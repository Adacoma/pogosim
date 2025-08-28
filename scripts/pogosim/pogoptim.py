#!/usr/bin/env python3
"""
Pogoptim – black‑box optimizer for Pogosim controllers

This script optimizes controller parameters defined in a multi‑value YAML
configuration (like `pogobatch`) using a pluggable optimizer (CMA‑ES or
Random Search). Each candidate genotype is **evaluated across the same
batch_options grid** (e.g., multiple arenas) and a given number of runs
per combination, exactly like pogobatch would do.

The objective/fitness is computed by a **user‑provided Python script** that
exposes a function (default name: `compute_objective`) taking the aggregated
results DataFrame from all runs+combinations of *one* genotype and returning
ONE float (higher is better by convention).

Outputs after the optimization:
  1) A PNG line plot of best‑so‑far fitness vs evaluation index.
  2) A YAML configuration file based on the original, but with the **best‑ever
     parameter values written in place** (and with no `genotype` nor
     `optimization_domain` entries).
  3) A Feather file with the **aggregated evaluation results DataFrame of the
     best individual**.

---
YAML expectations
-----------------
Instead of a top-level `genotype`, **attach an `optimization_domain` sub-entry directly to each config entry you want to optimize** (just like `batch_options`). Use `default_option` for the value that should be used by tools that don’t batch/optimize.

Example:

```yaml
# ... normal simulator config ...
controller:
  gains:
    k_p:
      default_option: 0.8
      optimization_domain: {type: float, min: 0.0, max: 5.0, init: 0.8}
    k_i:
      default_option: 0.0
      optimization_domain: {type: float, min: 0.0, max: 1.0}
  bias:
    default_option: 0.0
    optimization_domain: {type: float, min: -0.5, max: 0.5}

# vary arenas as usual (used by pogobatch grid)
environment:
  arena_file:
    default_option: arenas/disk.csv
    batch_options: [arenas/disk.csv, arenas/arena8.csv, arenas/annulus.csv, arenas/star7.csv]
    # You may also optimize a categorical choice explicitly:
    # optimization_domain: {type: categorical, choices: [arenas/disk.csv, arenas/arena8.csv, ...]}
```

Supported domain types:
  - `float`: keys `min`, `max`; optional `init`, optional `log: true` (optimize in log-space)
  - `int`:   keys `min`, `max`; optional `init`
  - `categorical`: key `choices: [...]`. **If `type` is omitted and the node has `batch_options`, we treat it as categorical over `batch_options`.**

`optimization_domain` keys are **stripped before each simulator run**. `default_option` is updated with the candidate value during optimization.

Objective script
----------------
----------------
Provide a Python file with (by default) a function:

```python
def compute_objective(df: "pandas.DataFrame") -> float:
    # return a *higher‑is‑better* score
    ...
```

CLI
---
Usage (typical):

    ./pogoptim.py -c multi.yaml -S ./pogosim -r 4 \
      -O fitness.py --optimizer cmaes --max-evals 80 --sigma0 0.3 \
      -o out/optim_exp

"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import logging
import warnings
import math
import os
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("pogoptim")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as paw
import yaml

from locomotion import compute_msd_per_agent
from pogobatch import PogobotBatchRunner, set_in_dict  # type: ignore

# ----------------------------------------------------------------------------
# Default objective: mean MSD across runs+arenas
# ----------------------------------------------------------------------------

def default_objective_mean_msd(df: pd.DataFrame) -> float:
    """Return the mean of per-agent MSD across *all* runs and arenas.
    If the input frame has no valid rows, returns -inf so optimizers will discard it."""
    #logger.debug("Columns available for objective: %s", list(df.columns))
    msd_df = compute_msd_per_agent(df)
    if msd_df.empty:
        logger.warning("Default MSD objective: empty input produced no MSD values; returning -inf")
        return float('-inf')
    return float(msd_df['MSD'].mean())


# ----------------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------------

def init_logging(verbose: bool) -> logging.Logger:
    """Configure the 'pogoptim' logger (no root usage, no duplicates)."""
    level = logging.DEBUG if verbose else logging.INFO

    # Configure our logger only
    logger.setLevel(level)
    logger.propagate = False  # CRITICAL: prevents duplicates via root

    # Ensure exactly one handler with a formatter
    for h in list(logger.handlers):
        logger.removeHandler(h)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', '%H:%M:%S'))
    handler.setLevel(level)
    logger.addHandler(handler)

    # Quiet dependencies (these affect their own named loggers, not root)
    logging.getLogger("pogobatch").setLevel(logging.WARNING if not verbose else logging.INFO)
    logging.getLogger("cma").setLevel(logging.CRITICAL)
    logging.getLogger("pyarrow").setLevel(logging.ERROR)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)

    # Optional: silence cma's 1-D warning banner
    warnings.filterwarnings("ignore", category=UserWarning, module=r"cma\..*")
    warnings.filterwarnings("ignore", message=r".*Optimization in 1-D is poorly tested.*")

    return logger

def _find_dotted_paths_for_key(node: Any, key: str, prefix: str = "") -> List[str]:
    """Recursively collect dotted paths whose *leaf key* equals `key`."""
    found: List[str] = []
    if isinstance(node, dict):
        for k, v in node.items():
            dotted = f"{prefix}.{k}" if prefix else k
            if k == key:
                found.append(dotted)
            found.extend(_find_dotted_paths_for_key(v, key, dotted))
    elif isinstance(node, list):
        # Do not dive into plain lists (they are not dot-addressable in config semantics)
        pass
    return found

def load_objective(path: Optional[str], func_name: str = "compute_objective"):
    # If no objective script provided, fall back to default MSD-based fitness
    if path is None or str(path).strip() == "":
        logger.warning("No objective script provided; using DEFAULT fitness: mean MSD across runs and arenas.")
        return default_objective_mean_msd

    spec = importlib.util.spec_from_file_location("pogoptim_user_objective", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import objective from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    if not hasattr(mod, func_name):
        raise AttributeError(f"Objective function '{func_name}' not found in {path}")
    return getattr(mod, func_name)


@dataclass
class VarSpec:
    path: str
    kind: str  # float|int|categorical
    lo: Optional[float] = None
    hi: Optional[float] = None
    log: bool = False
    choices: Optional[List[Any]] = None
    init: Optional[float] = None  # number or index for categorical


def discover_optimization_domains(config: Dict[str, Any]) -> List[VarSpec]:
    specs: List[VarSpec] = []

    def rec(node: Any, dotted: str) -> None:
        if isinstance(node, dict):
            # If this node declares an optimization domain, register a variable bound to this node path
            if "optimization_domain" in node:
                dom = node.get("optimization_domain", {}) or {}
                kind = dom.get("type")
                # Fallback to categorical if choices present or batch_options exist
                choices = dom.get("choices") if isinstance(dom, dict) else None
                if (not kind) and (choices is not None):
                    kind = "categorical"
                if (not kind) and ("batch_options" in node):
                    kind = "categorical"
                    choices = node.get("batch_options")

                if kind == "categorical":
                    if not choices or not isinstance(choices, list):
                        raise RuntimeError(f"categorical domain for {dotted} needs a non-empty 'choices' list (or 'batch_options').")
                    init_val = dom.get("init", node.get("default_option"))
                    if init_val in choices:
                        init_idx = choices.index(init_val)
                    else:
                        init_idx = 0
                    specs.append(VarSpec(path=dotted, kind="categorical", choices=choices, init=float(init_idx)))
                elif kind in ("float", "int"):
                    try:
                        lo = float(dom.get("min"))
                        hi = float(dom.get("max"))
                    except Exception as exc:
                        raise RuntimeError(f"Invalid [min,max] for {dotted}") from exc
                    if not math.isfinite(lo) or not math.isfinite(hi) or not (lo < hi):
                        raise RuntimeError(f"Invalid [min,max] for {dotted}")
                    log_flag = bool(dom.get("log", False))
                    init = dom.get("init", node.get("default_option"))
                    init = None if init is None else float(init)
                    specs.append(VarSpec(path=dotted, kind=kind, lo=lo, hi=hi, log=log_flag, init=init))
                else:
                    raise RuntimeError(f"Unsupported or missing domain type for {dotted}")

            # Recurse into children
            for k, v in node.items():
                if k == "optimization_domain":
                    continue
                newdot = f"{dotted}.{k}" if dotted else k
                rec(v, newdot)
        # Ignore lists/scalars for domain discovery

    rec(config, "")
    if not specs:
        raise RuntimeError("Config contains no 'optimization_domain' entries to optimize.")
    return specs


def strip_optimization_domains(config: Dict[str, Any]) -> Dict[str, Any]:
    cfg = copy.deepcopy(config)

    def rec(node: Any) -> None:
        if isinstance(node, dict):
            if "optimization_domain" in node:
                del node["optimization_domain"]
            for v in node.values():
                rec(v)
        elif isinstance(node, list):
            for it in node:
                rec(it)
    rec(cfg)
    return cfg


def encode_x0(specs: List[VarSpec]) -> np.ndarray:
    xs: List[float] = []
    for s in specs:
        if s.kind == "categorical":
            xs.append(float(0 if s.init is None else s.init))
        elif s.kind == "int":
            if s.init is None:
                xs.append((s.lo + s.hi) * 0.5)
            else:
                xs.append(float(s.init))
        else:  # float
            if s.init is None:
                val = (s.lo + s.hi) * 0.5
            else:
                val = float(s.init)
            xs.append(math.log(val) if s.log else val)
    return np.array(xs, dtype=float)


def clamp_and_decode(specs: List[VarSpec], x: np.ndarray) -> Dict[str, Any]:
    assert len(specs) == len(x)
    out: Dict[str, Any] = {}
    for s, v in zip(specs, x):
        if s.kind == "categorical":
            idx = int(round(v))
            idx = max(0, min(len(s.choices) - 1, idx))
            out[s.path] = s.choices[idx]
        elif s.kind == "int":
            vv = int(round(v))
            vv = int(max(s.lo, min(s.hi, vv)))
            out[s.path] = vv
        else:  # float
            vv = float(v)
            if s.log:
                vv = math.exp(vv)
            vv = max(s.lo, min(s.hi, vv))
            out[s.path] = vv
    return out


def _resolve_node(cfg: Dict[str, Any], dotted: str) -> Any:
    parts = dotted.split('.') if dotted else []
    node: Any = cfg
    for p in parts:
        if not isinstance(node, dict) or p not in node:
            raise KeyError(f"Path not found: {dotted}")
        node = node[p]
    return node


def set_optimized_values_in_config(base_cfg: Dict[str, Any], values: Dict[str, Any]) -> Dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    for dotted, val in values.items():
        try:
            node = _resolve_node(cfg, dotted)
            if isinstance(node, dict):
                # Prefer updating default_option for compatibility with pogobatch
                node["default_option"] = val
            else:
                # Fallback: directly set the scalar value
                set_in_dict(cfg, dotted, val)
        except KeyError:
            # Create a small node with default_option if the path is missing
            set_in_dict(cfg, dotted, {"default_option": val})
    return cfg


def write_yaml(obj: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def run_evaluation(
    cfg_for_eval: Dict[str, Any],
    simulator_binary: str,
    runs: int,
    temp_base: str,
    backend: str,
    keep_temp: bool,
    retries: int,
) -> pd.DataFrame:
    # Use a per-evaluation temp area for batch outputs
    os.makedirs(temp_base, exist_ok=True)
    eval_tmp = tempfile.mkdtemp(prefix="eval_", dir=temp_base)

    try:
        # Ensure result_new_columns contains a usable path to arena_file
        # (pogobatch expects dotted *paths* there)
        # This mirrors pogobatch's feature and works even if arena_file is nested,
        # e.g. under "environment.arena_file".
        cfg_for_eval = copy.deepcopy(cfg_for_eval)
        rnc: List[str] = list(cfg_for_eval.get("result_new_columns", []) or [])
        wants_arena_basename = "arena_file" in rnc
        # If the user asked for arena_file (basename) but it's nested, append its dotted path
        arena_paths = _find_dotted_paths_for_key(cfg_for_eval, "arena_file")
        # Prefer the shortest dotted candidate to avoid overly verbose column names
        arena_path = sorted(arena_paths, key=len)[0] if arena_paths else None
        if arena_path and (wants_arena_basename or "arena_file" not in cfg_for_eval):
            if arena_path not in rnc:
                rnc.append(arena_path)
        # Also auto-add when user forgot result_new_columns entirely
        if not wants_arena_basename and arena_path and ("arena_file" not in rnc):
            rnc.append(arena_path)
        if rnc:
            cfg_for_eval["result_new_columns"] = rnc

        # Persist config (batch runner reads from a file)
        cfg_path = os.path.join(eval_tmp, "multi.yaml")
        write_yaml(cfg_for_eval, cfg_path)

        os.makedirs(os.path.join(eval_tmp, "tmp"), exist_ok=True)
        os.makedirs(os.path.join(eval_tmp, "out"), exist_ok=True)

        runner = PogobotBatchRunner(
            multi_config_file=cfg_path,
            runs=runs,
            simulator_binary=simulator_binary,
            temp_base=os.path.join(eval_tmp, "tmp"),
            output_dir=os.path.join(eval_tmp, "out"),
            backend=backend,
            keep_temp=keep_temp,
            verbose=False,
            retries=retries,
        )

        # Be quiet unless -v: temporarily raise root level to WARNING while pogobatch runs.
        prev_level = logger.level
        if prev_level > logging.INFO:
            logger.setLevel(logging.WARNING)
        try:
            outputs = runner.run_all()  # list of feather paths; usually one aggregated
        finally:
            logger.setLevel(prev_level)

        if not outputs:
            raise RuntimeError("No output files produced by batch runner.")
        # Merge all produced output files (defensive; often only one file exists)
        dfs = [pd.read_feather(p) for p in outputs if os.path.exists(p)]
        if not dfs:
            raise RuntimeError("Produced output files are missing or unreadable.")

        df = pd.concat(dfs, ignore_index=True)
        # Normalize column name back to "arena_file" if pogobatch created a dotted one.
        for c in list(df.columns):
            if c.endswith(".arena_file") and "arena_file" not in df.columns:
                df = df.rename(columns={c: "arena_file"})
            elif c.endswith(".arena_file") and "arena_file" in df.columns:
                df = df.drop(columns=[c])
        return df
    finally:
        if not keep_temp:
            shutil.rmtree(eval_tmp, ignore_errors=True)


# ----------------------------------------------------------------------------
# Optimizers
# ----------------------------------------------------------------------------

class BaseOptimizer:
    def __init__(self, dim: int):
        self.dim = dim
        self.best_f = -np.inf
        self.best_x = None  # type: Optional[np.ndarray]

    def run(self, ask_tell_loop):
        raise NotImplementedError


class RandomSearch(BaseOptimizer):
    def __init__(self, dim: int, bounds: List[Tuple[float, float]], max_evals: int, seed: int = 42):
        super().__init__(dim)
        self.bounds = bounds
        self.max_evals = max_evals
        self.rng = np.random.default_rng(seed)

    def run(self, ask_tell_loop):
        evals = 0
        while evals < self.max_evals:
            x = np.array([self.rng.uniform(lo, hi) for (lo, hi) in self.bounds], dtype=float)
            f = ask_tell_loop(x)
            if f > self.best_f:
                self.best_f = f
                self.best_x = x.copy()
            evals += 1
        logger.info("random: evals=%d  best=%.6g", evals, self.best_f)



class CMAES(BaseOptimizer):
    def __init__(self, dim: int, x0: np.ndarray, sigma0: float, popsize: Optional[int], seed: int, max_evals: int):
        super().__init__(dim)
        try:
            import cma  # pyright: ignore[reportMissingImports]
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("CMA‑ES optimizer requested but 'cma' package is not available.") from exc
        self.cma = cma
        opts = {
            'seed': seed or 0,
            'verb_disp': 0,   # no iteration display
            'verb_log': 0,    # no cma.log, outcma.* files
            'verbose': -9,    # ultra quiet
        }
        if popsize:
            opts["popsize"] = int(popsize)
        self.es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
        self.max_evals = max_evals

    def run(self, ask_tell_loop):
        evals = 0
        gen_idx = 0
        while not self.es.stop():
            xs = self.es.ask()
            fs = []
            pop_f = []
            for x in xs:
                f = ask_tell_loop(np.asarray(x, dtype=float))
                fs.append(-float(f))  # CMA‑ES minimizes; we maximize
                pop_f.append(float(f))
                if f > self.best_f:
                    self.best_f = f
                    self.best_x = np.array(x, dtype=float)
                evals += 1
                if evals >= self.max_evals:
                    break
            self.es.tell(xs[: len(fs)], fs)
            # Per-generation one-liner (quiet unless -v flipped on globally)
            if pop_f:
                gen_idx += 1
                gmax = float(np.max(pop_f))
                gmin = float(np.min(pop_f))
                gmean = float(np.mean(pop_f))
                logger.info("gen %03d: pop=%d  f[best/mean/min]=[%.6g/%.6g/%.6g]  best_so_far=%.6g",
                             gen_idx, len(pop_f), gmax, gmean, gmin, self.best_f)
            if evals >= self.max_evals:
                break
        # finalize
        try:
            self.es.result  # force finalize internally
        except Exception:
            pass


# ----------------------------------------------------------------------------
# Main optimization driver
# ----------------------------------------------------------------------------

def optimize(
    multi_config_path: str,
    simulator_binary: str,
    objective_path: Optional[str],
    objective_func_name: str,
    runs: int,
    temp_base: str,
    backend: str,
    keep_temp: bool,
    retries: int,
    optimizer_name: str,
    max_evals: int,
    sigma0: float,
    popsize: Optional[int],
    seed: int,
    out_dir: str,
):
    os.makedirs(out_dir, exist_ok=True)

    # Load config and discover per-node optimization domains
    with open(multi_config_path, "r", encoding="utf-8") as f:
        full_cfg = yaml.safe_load(f)
    specs = discover_optimization_domains(full_cfg)
    base_cfg = strip_optimization_domains(full_cfg)

    # Objective
    objective_fn = load_objective(objective_path, objective_func_name)

    # Vectorization helpers
    x0 = encode_x0(specs)

    # Build bounds in *internal* space (log applied already for float with log)
    bounds: List[Tuple[float, float]] = []
    for s, v0 in zip(specs, x0):
        if s.kind == "categorical":
            bounds.append((0.0, float(len(s.choices) - 1)))
        elif s.kind == "int":
            bounds.append((float(s.lo), float(s.hi)))
        else:  # float
            if s.log:
                bounds.append((math.log(s.lo), math.log(s.hi)))
            else:
                bounds.append((float(s.lo), float(s.hi)))

    # History
    history_rows = []  # each: dict with eval, fitness, best_so_far, x(json), values(json)
    best_f = -np.inf
    best_values: Optional[Dict[str, Any]] = None
    best_df: Optional[pd.DataFrame] = None

    def eval_one(x_internal: np.ndarray) -> float:
        nonlocal best_f, best_values, best_df
        # Clamp+decode to dict of dotted→value
        values = clamp_and_decode(specs, x_internal)
        # Build per‑eval config (set genotype values)
        cfg_eval = set_optimized_values_in_config(base_cfg, values)
        # Evaluate using batch runner
        df = run_evaluation(
            cfg_for_eval=cfg_eval,
            simulator_binary=simulator_binary,
            runs=runs,
            temp_base=temp_base,
            backend=backend,
            keep_temp=keep_temp,
            retries=retries,
        )
        # Fitness
        try:
            fitness = float(objective_fn(df))
        except Exception as exc:  # safeguard
            logger.error("Objective raised %s; returning -inf", exc)
            fitness = float("-inf")
        # Update history
        now = time.time()
        nonlocal_eval_idx = len(history_rows) + 1
        best_f = max(best_f, fitness)
        if fitness >= best_f:
            best_values = values
            best_df = df
        history_rows.append({
            "eval": nonlocal_eval_idx,
            "fitness": fitness,
            "best_so_far": best_f,
            "x_internal": json.dumps(list(map(float, x_internal))),
            "values": json.dumps(values),
            "timestamp": now,
        })
        #logger.debug("Eval %d → fitness=%.6g | best=%.6g", nonlocal_eval_idx, fitness, best_f)
        logger.debug(f"Eval %d → fitness=%.6g | best=%.6g (genotype: {x_internal})", nonlocal_eval_idx, fitness, best_f)
        return fitness

    # Choose optimizer
    if optimizer_name.lower() == "random":
        opt = RandomSearch(dim=len(specs), bounds=bounds, max_evals=max_evals, seed=seed)
    elif optimizer_name.lower() == "cmaes":
        opt = CMAES(dim=len(specs), x0=x0, sigma0=sigma0, popsize=popsize, seed=seed, max_evals=max_evals)
    else:
        raise RuntimeError(f"Unknown optimizer: {optimizer_name}")

    # Run optimization
    logger.info("Starting optimization: %s | dim=%d | max_evals=%d", optimizer_name.upper(), len(specs), max_evals)
    opt.run(eval_one)

    # Save history
    hist_df = pd.DataFrame(history_rows)
    hist_csv = os.path.join(out_dir, "opt_history.csv")
    hist_df.to_csv(hist_csv, index=False)

    # Plot best‑so‑far
    fig = plt.figure(figsize=(8, 4.5))
    ax = fig.add_subplot(111)
    ax.plot(hist_df["eval"], hist_df["best_so_far"], label="best so far")
    ax.set_xlabel("evaluation")
    ax.set_ylabel("fitness")
    ax.grid(True, alpha=0.3)
    ax.legend()
    png_path = os.path.join(out_dir, "fitness_vs_eval.png")
    fig.tight_layout()
    fig.savefig(png_path, dpi=144)
    plt.close(fig)

    # Write best config
    if best_values is None:
        raise RuntimeError("No successful evaluations; cannot produce best config/results.")

    best_cfg = set_optimized_values_in_config(base_cfg, best_values)
    # Strip any leftover optimization_domain keys and write
    best_cfg = strip_optimization_domains(best_cfg)
    best_yaml = os.path.join(out_dir, "best_config.yaml")
    write_yaml(best_cfg, best_yaml)

    # Save best results feather with schema metadata including configuration text
    if best_df is None:
        raise RuntimeError("Internal error: best_df is None despite best_values present.")
    table = pa.Table.from_pandas(best_df)
    with open(best_yaml, "r", encoding="utf-8") as f:
        cfg_text = f.read()
    table = table.replace_schema_metadata({b"configuration": cfg_text.encode("utf-8")})
    best_feather = os.path.join(out_dir, "best_results.feather")
    paw.write_feather(table, best_feather)

    # Summary JSON
    summary = {
        "optimizer": optimizer_name,
        "max_evals": max_evals,
        "best_fitness": float(np.max(hist_df["best_so_far"])) if len(hist_df) else None,
        "best_values": best_values,
        "files": {
            "history_csv": hist_df.shape[0] and hist_csv,
            "plot_png": png_path,
            "best_config_yaml": best_yaml,
            "best_results_feather": best_feather,
        },
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("Done. Best fitness: %.6g", summary["best_fitness"])
    logger.info("Best values: %s", json.dumps(best_values))
    #logger.info("Artifacts in %s", out_dir)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Optimize Pogosim controllers over batch_options + runs using CMA‑ES or Random Search. Annotate any config entry to optimize by adding an 'optimization_domain' sub-entry.")
    p.add_argument("-c", "--config", required=True, help="Path to multi-value YAML (entries to optimize must have 'optimization_domain' sub-entries).")
    p.add_argument("-S", "--simulator-binary", required=True, help="Path to pogosim executable.")
    p.add_argument("-r", "--runs", type=int, default=1, help="Number of runs per batch combination (default: 1).")
    p.add_argument("-t", "--temp-base", default="tmp_opt", help="Base temp directory (default: tmp_opt).")
    p.add_argument("-o", "--output-dir", default="opt_out", help="Directory to store optimization outputs (default: opt_out).")
    p.add_argument("-O", "--objective", required=False, default=None, help="Optional path to Python file exposing the fitness function. If omitted, DEFAULT fitness is used: mean MSD across runs and arenas.")
    p.add_argument("--objective-func", default="compute_objective", help="Function name inside --objective (default: compute_objective).")
    p.add_argument("--optimizer", choices=["cmaes", "random"], default="cmaes", help="Optimizer to use (default: cmaes).")
    p.add_argument("--max-evals", type=int, default=50, help="Maximum number of objective evaluations (default: 50).")
    p.add_argument("--sigma0", type=float, default=0.3, help="Initial CMA‑ES sigma (default: 0.3).")
    p.add_argument("--popsize", type=int, default=None, help="CMA‑ES population size override (optional).")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    p.add_argument("--backend", choices=["multiprocessing", "ray"], default="multiprocessing", help="Parallel backend for *batch evaluation* (default: multiprocessing).")
    p.add_argument("--keep-temp", action="store_true", help="Keep per‑eval temp directories (useful for debugging).")
    p.add_argument("-R", "--retries", type=int, default=5, help="How many times to relaunch a run upon simulator crash (default: 5).")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    init_logging(args.verbose)

    try:
        optimize(
            multi_config_path=args.config,
            simulator_binary=args.simulator_binary,
            objective_path=args.objective,
            objective_func_name=args.objective_func,
            runs=args.runs,
            temp_base=args.temp_base,
            backend=args.backend,
            keep_temp=args.keep_temp,
            retries=args.retries,
            optimizer_name=args.optimizer,
            max_evals=args.max_evals,
            sigma0=args.sigma0,
            popsize=args.popsize,
            seed=args.seed,
            out_dir=args.output_dir,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Fatal: %s", exc)
        sys.exit(2)


if __name__ == "__main__":
    main()


# MODELINE "{{{1
# vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
# vim:foldmethod=marker
