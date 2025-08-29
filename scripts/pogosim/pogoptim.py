#!/usr/bin/env python3
"""
Pogoptim – black-box & quality-diversity optimizer for Pogosim controllers

Changes vs previous version:
  • Internal search space is now NORMALIZED to [0,1]^d for every variable
    (float, int, categorical). This makes sigma0/popsize scale-agnostic.
  • Optional QD optimizer (MAP-Elites via QDpy). If chosen, we build and
    return a repertoire/container of elites instead of a single best.
  • Default QD feature descriptors (when no objective script is provided):
      ( max per-agent MSD over all runs+arenas,
        std deviation of per-agent MSD over all runs+arenas )
  • Objective function compatibility:
      - If your objective returns a scalar → we’ll compute default QD features.
      - Optionally, your objective may return (fitness, features) where
        `features` is a 1-D list/tuple/np.array of floats.
"""

from __future__ import annotations

from multiprocessing import current_process
import argparse
import copy
import importlib.util
import json
import logging
import warnings
import math
import os
import pickle
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Sequence
import traceback

logger = logging.getLogger("pogoptim")
_worker_logging_inited = False  # module-level flag

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as paw
import yaml

from locomotion import compute_msd_per_agent
from pogobatch import PogobotBatchRunner, set_in_dict  # type: ignore

# ----------------------------------------------------------------------------
# Default objective and QD features
# ----------------------------------------------------------------------------

def fd_polar_order_phi(df: pd.DataFrame) -> float:
    """Mean global polar order Φ in [0,1] over time."""
    g = df[df.get("robot_category", "robots") == "robots"][["time", "angle"]].dropna()
    if g.empty:
        return 0.0
    g = g.assign(c=np.cos(g["angle"].to_numpy()),
                 s=np.sin(g["angle"].to_numpy()))
    per_t = g.groupby("time")[["c", "s"]].mean()
    phi_t = np.sqrt(per_t["c"]**2 + per_t["s"]**2)
    return float(np.clip(phi_t.mean(), 0.0, 1.0))

def fd_straightness(df: pd.DataFrame) -> float:
    """Mean straightness S in [0,1] across tracks (run, arena_file, robot_id)."""
    cols = ["run", "arena_file", "robot_id", "time", "x", "y", "robot_category"]
    g = df[cols].copy()
    g = g[g["robot_category"] == "robots"].dropna(subset=["time", "x", "y"])
    if g.empty:
        return 0.0

    def one_track(track: pd.DataFrame) -> float:
        t = track.sort_values("time")
        if len(t) < 2:
            return 0.0
        xy = t[["x", "y"]].to_numpy()
        steps = np.sqrt(((xy[1:] - xy[:-1]) ** 2).sum(axis=1))
        path_len = float(steps.sum())
        if path_len <= 1e-12:
            return 0.0
        disp = float(np.linalg.norm(xy[-1] - xy[0]))
        return float(np.clip(disp / path_len, 0.0, 1.0))

    # New pandas (>=2.2): exclude grouping columns from the DataFrame seen by apply
    try:
        s_vals = g.groupby(["run", "arena_file", "robot_id"], sort=False).apply(
            one_track, include_groups=False
        )
    except TypeError:
        # Older pandas: explicitly select only the columns the function needs
        s_vals = (
            g.groupby(["run", "arena_file", "robot_id"], sort=False)
             .apply(lambda t: one_track(t[["time", "x", "y"]]))
        )

    return float(np.clip(s_vals.mean() if len(s_vals) else 0.0, 0.0, 1.0))

def default_qd_features_unit(df: pd.DataFrame) -> np.ndarray:
    """2-D features in [0,1]^2: (Φ, S)."""
    return np.array([fd_polar_order_phi(df), fd_straightness(df)], dtype=float)

def default_objective_mean_msd(df: pd.DataFrame) -> float:
    """Return the mean of per-agent MSD across *all* runs and arenas."""
    msd_df = compute_msd_per_agent(df)
    if msd_df.empty:
        logger.warning("Default MSD objective: empty input produced no MSD values; returning -inf")
        return float('-inf')
    return float(msd_df['MSD'].mean())


def default_qd_features_maxstd_msd(df: pd.DataFrame) -> np.ndarray:
    """Return default 2-D features for QD: (polar order, straightness)."""
    msd_df = compute_msd_per_agent(df)
    if msd_df.empty:
        return np.array([0.0, 0.0], dtype=float)
    vals = np.asarray(msd_df["MSD"], dtype=float)
    return np.array([float(np.max(vals)), float(np.std(vals, ddof=0))], dtype=float)


# ----------------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------------

def init_logging(verbose: bool) -> logging.Logger:
    level = logging.DEBUG if verbose else logging.INFO
    logger.setLevel(level)
    logger.propagate = False
    for h in list(logger.handlers):
        logger.removeHandler(h)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', '%H:%M:%S'))
    handler.setLevel(level)
    logger.addHandler(handler)

    logging.getLogger("pogobatch").setLevel(logging.WARNING if not verbose else logging.INFO)
    logging.getLogger("cma").setLevel(logging.CRITICAL)
    logging.getLogger("pyarrow").setLevel(logging.ERROR)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)

    warnings.filterwarnings("ignore", category=UserWarning, module=r"cma\..*")
    warnings.filterwarnings("ignore", message=r".*Optimization in 1-D is poorly tested.*")
    return logger

def _init_worker_logging_quiet():
    """Ensure child processes don't chat unless asked."""
    global _worker_logging_inited
    if _worker_logging_inited:
        return
    from multiprocessing import current_process
    if current_process().name != "MainProcess":
        # Our logger
        logger.setLevel(logging.WARNING)
        logger.propagate = False
        for h in list(logger.handlers):
            logger.removeHandler(h)
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', '%H:%M:%S'))
        h.setLevel(logging.WARNING)
        logger.addHandler(h)
        # Third-party loggers that sometimes get chatty in workers
        logging.getLogger("pogobatch").setLevel(logging.WARNING)
        logging.getLogger("pyarrow").setLevel(logging.ERROR)
        logging.getLogger("matplotlib").setLevel(logging.WARNING)
        logging.getLogger("PIL").setLevel(logging.WARNING)
    _worker_logging_inited = True


# ----------------------------------------------------------------------------
# Config utilities
# ----------------------------------------------------------------------------

def _find_dotted_paths_for_key(node: Any, key: str, prefix: str = "") -> List[str]:
    found: List[str] = []
    if isinstance(node, dict):
        for k, v in node.items():
            dotted = f"{prefix}.{k}" if prefix else k
            if k == key:
                found.append(dotted)
            found.extend(_find_dotted_paths_for_key(v, key, dotted))
    return found


def load_objective(path: Optional[str], func_name: str = "compute_objective"):
    if path is None or str(path).strip() == "":
        # Warn ONLY from the main process to avoid duplicates with workers
        if current_process().name == "MainProcess":
            logger.warning("No objective script provided; DEFAULT fitness = mean MSD (features default to [polar order, straightness]).")
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
            if "optimization_domain" in node:
                dom = node.get("optimization_domain", {}) or {}
                kind = dom.get("type")
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
                    init_idx = choices.index(init_val) if init_val in choices else 0
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
            for k, v in node.items():
                if k == "optimization_domain":
                    continue
                newdot = f"{dotted}.{k}" if dotted else k
                rec(v, newdot)
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


# ----------------------------------------------------------------------------
# NEW: normalized encoding (internal space u in [0,1])
# ----------------------------------------------------------------------------

def _encode_to_unit(spec: VarSpec, val: Optional[float]) -> float:
    if spec.kind == "categorical":
        n = len(spec.choices)
        idx = 0 if val is None else int(round(val))
        return 0.0 if n <= 1 else max(0.0, min(1.0, idx / (n - 1)))
    if spec.kind == "int":
        mid = (spec.lo + spec.hi) * 0.5 if val is None else float(val)
        return (mid - spec.lo) / (spec.hi - spec.lo)
    # float
    if val is None:
        v = (spec.lo + spec.hi) * 0.5
    else:
        v = float(val)
    if spec.log:
        vlo = math.log(spec.lo)
        vhi = math.log(spec.hi)
        return (math.log(v) - vlo) / (vhi - vlo)
    else:
        return (v - spec.lo) / (spec.hi - spec.lo)


def encode_x0_unit(specs: List[VarSpec]) -> np.ndarray:
    xs = []
    for s in specs:
        xs.append(_encode_to_unit(s, s.init))
    return np.asarray(xs, dtype=float)


def _decode_from_unit(spec: VarSpec, u: float) -> Any:
    uu = max(0.0, min(1.0, float(u)))
    if spec.kind == "categorical":
        n = len(spec.choices)
        idx = 0 if n <= 1 else int(round(uu * (n - 1)))
        idx = max(0, min(n - 1, idx))
        return spec.choices[idx]
    if spec.kind == "int":
        v = spec.lo + uu * (spec.hi - spec.lo)
        return int(max(spec.lo, min(spec.hi, round(v))))
    # float
    if spec.log:
        vlo = math.log(spec.lo)
        vhi = math.log(spec.hi)
        v = math.exp(vlo + uu * (vhi - vlo))
    else:
        v = spec.lo + uu * (spec.hi - spec.lo)
    return float(max(spec.lo, min(spec.hi, v)))


def decode_unit_vector(specs: List[VarSpec], u: np.ndarray) -> Dict[str, Any]:
    assert len(specs) == len(u)
    values: Dict[str, Any] = {}
    for s, x in zip(specs, u):
        values[s.path] = _decode_from_unit(s, float(x))
    return values


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
                node["default_option"] = val
            else:
                set_in_dict(cfg, dotted, val)
        except KeyError:
            set_in_dict(cfg, dotted, {"default_option": val})
    return cfg


def write_yaml(obj: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


# ----------------------------------------------------------------------------
# Run a single evaluation (batch grid x runs) and compute fitness/features
# ----------------------------------------------------------------------------

def run_evaluation(
    cfg_for_eval: Dict[str, Any],
    simulator_binary: str,
    runs: int,
    temp_base: str,
    backend: str,
    keep_temp: bool,
    retries: int,
) -> pd.DataFrame:
    os.makedirs(temp_base, exist_ok=True)
    eval_tmp = tempfile.mkdtemp(prefix="eval_", dir=temp_base)
    try:
        cfg_for_eval = copy.deepcopy(cfg_for_eval)
        rnc: List[str] = list(cfg_for_eval.get("result_new_columns", []) or [])
        wants_arena_basename = "arena_file" in rnc
        arena_paths = _find_dotted_paths_for_key(cfg_for_eval, "arena_file")
        arena_path = sorted(arena_paths, key=len)[0] if arena_paths else None
        if arena_path and (wants_arena_basename or "arena_file" not in cfg_for_eval):
            if arena_path not in rnc:
                rnc.append(arena_path)
        if not wants_arena_basename and arena_path and ("arena_file" not in rnc):
            rnc.append(arena_path)
        if rnc:
            cfg_for_eval["result_new_columns"] = rnc

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
        prev_level = logger.level
        if prev_level > logging.INFO:
            logger.setLevel(logging.WARNING)
        try:
            outputs = runner.run_all()
        finally:
            logger.setLevel(prev_level)

        if not outputs:
            raise RuntimeError("No output files produced by batch runner.")
        dfs = [pd.read_feather(p) for p in outputs if os.path.exists(p)]
        if not dfs:
            raise RuntimeError("Produced output files are missing or unreadable.")
        df = pd.concat(dfs, ignore_index=True)

        for c in list(df.columns):
            if c.endswith(".arena_file") and "arena_file" not in df.columns:
                df = df.rename(columns={c: "arena_file"})
            elif c.endswith(".arena_file") and "arena_file" in df.columns:
                df = df.drop(columns=[c])
        return df
    finally:
        if not keep_temp:
            shutil.rmtree(eval_tmp, ignore_errors=True)


def eval_fitness_and_features(
    u_internal: np.ndarray,
    specs: List[VarSpec],
    base_cfg: Dict[str, Any],
    simulator_binary: str,
    runs: int,
    temp_base: str,
    backend: str,
    keep_temp: bool,
    retries: int,
    objective_fn,
    objective_returns_features: bool,
    default_features_fn,
) -> Tuple[float, np.ndarray, pd.DataFrame, Dict[str, Any]]:
    PENALTY = -1e9  # finite big negative fitness

    values = decode_unit_vector(specs, u_internal)
    cfg_eval = set_optimized_values_in_config(base_cfg, values)

    try:
        df = run_evaluation(
            cfg_for_eval=cfg_eval,
            simulator_binary=simulator_binary,
            runs=runs,
            temp_base=temp_base,
            backend=backend,
            keep_temp=keep_temp,
            retries=retries,
        )
    except Exception as exc:
        logger.error("Evaluation error: %s; penalizing candidate.", exc)
        return PENALTY, np.zeros(2, dtype=float), pd.DataFrame(), values

    fitness = None
    features = None
    try:
        out = objective_fn(df)
        if isinstance(out, (tuple, list)) and len(out) >= 2:
            fitness = float(out[0])
            feats = out[1]
            if isinstance(feats, dict):
                feats = list(feats.values())
            features = np.asarray(feats, dtype=float).ravel()
        else:
            fitness = float(out)
    except Exception as exc:
        logger.error("Objective error: %s; penalizing candidate.", exc)
        fitness = PENALTY

    # Default features if none provided
    if features is None:
        try:
            features = np.asarray(default_features_fn(df), dtype=float).ravel()
        except Exception:
            features = np.zeros(2, dtype=float)

    # Clamp to finite values
    if not np.isfinite(fitness):
        fitness = PENALTY
    if not np.all(np.isfinite(features)):
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    return float(fitness), features, df, values



# ----------------------------------------------------------------------------
# Optimizers (Random, CMA-ES)
# ----------------------------------------------------------------------------

class BaseOptimizer:
    def __init__(self, dim: int):
        self.dim = dim
        self.best_f = -np.inf
        self.best_u = None  # type: Optional[np.ndarray]

    def run(self, ask_tell_loop):
        raise NotImplementedError

class RandomSearch(BaseOptimizer):
    def __init__(self, dim: int, max_evals: int, seed: int = 42):
        super().__init__(dim)
        self.max_evals = max_evals
        self.rng = np.random.default_rng(seed)

    def run(self, ask_tell_loop):
        evals = 0
        while evals < self.max_evals:
            u = self.rng.uniform(0.0, 1.0, size=self.dim).astype(float)
            f = ask_tell_loop(u)
            if f > self.best_f:
                self.best_f = f
                self.best_u = u.copy()
            evals += 1
        logger.info("random: evals=%d  best=%.6g", evals, self.best_f)


class CMAES(BaseOptimizer):
    def __init__(self, dim: int, u0: np.ndarray, sigma0: float,
                 popsize: Optional[int], seed: int, max_evals: int):
        super().__init__(dim)
        try:
            import cma  # type: ignore
        except Exception as exc:
            raise RuntimeError("CMA-ES requested but 'cma' package is not available.") from exc
        self.cma = cma

        # Robust popsize (never larger than budget; ≥1)
        default_pop = max(4, 4 + int(3 * math.log(max(dim, 1))))
        pop = int(popsize) if popsize else default_pop
        pop = max(1, min(pop, max_evals))

        opts = {
            "seed": seed or 0,
            #"bounds": [0.0, 1.0],   # normalized box. Can make CMA-ES crash on some versions
            "verb_disp": 0,
            "verb_log": 0,
            "verbose": -9,
            "popsize": pop,
        }
        self.es = cma.CMAEvolutionStrategy(u0.tolist(), float(sigma0), opts)
        self.max_evals = int(max_evals)
        self._pop = pop  # remember for ask(number=...)

    def run(self, ask_tell_loop):
        evals = 0
        gen_idx = 0

        while evals < self.max_evals:
            remaining = self.max_evals - evals
            n = int(min(self._pop, remaining))
            if n <= 0:
                break

            # Ask exactly n candidates
            try:
                xs = self.es.ask(number=n)
            except Exception:
                xs = self.es.ask()[:n]

            # IMPORTANT: clip candidates to [0,1]^d before evaluating
            xs_eval = [np.clip(np.asarray(u, float), 0.0, 1.0) for u in xs]

            fs = []
            pop_f = []
            for u_eval in xs_eval:
                f = ask_tell_loop(u_eval)
                if not np.isfinite(f):
                    f = -1e9  # finite penalty
                fs.append(-float(f))       # CMA minimizes
                pop_f.append(float(f))
                if f > self.best_f:
                    self.best_f = f
                    self.best_u = u_eval.copy()

            # Tell CMA the SAME points we actually evaluated (the clipped ones)
            self.es.tell([u.tolist() for u in xs_eval], fs)

            evals += n
            gen_idx += 1
            gmax = float(np.max(pop_f))
            gmin = float(np.min(pop_f))
            gmean = float(np.mean(pop_f))
            logger.info("gen %03d: pop=%d  f[best/mean/min]=[%.6g/%.6g/%.6g]  best_so_far=%.6g",
                        gen_idx, len(pop_f), gmax, gmean, gmin, self.best_f)

            # Stop if CMA says so or budget hit
            try:
                if self.es.stop():
                    break
            except Exception:
                break




# ----------------------------------------------------------------------------
# QDpy MAP-Elites driver
# ----------------------------------------------------------------------------

def run_qdpy_map_elites(
    specs: List[VarSpec],
    base_cfg: Dict[str, Any],
    simulator_binary: str,
    runs: int,
    temp_base: str,
    backend: str,
    keep_temp: bool,
    retries: int,
    objective_fn,
    default_features_fn,
    qd_shape: Sequence[int],
    qd_init_samples: int,
    qd_batch: int,
    max_evals: int,
    sigma0: float,
    seed: int,
    out_dir: str,
    qd_algo_kwargs: Optional[Dict[str, Any]] = None,   # <<< NEW
) -> Dict[str, Any]:
    from qdpy import algorithms, containers, plots, base
    rng = np.random.default_rng(seed)
    max_evals = int(max_evals)

    def evaluate_one(u_vec: np.ndarray):
        f, feats, df, _values = eval_fitness_and_features(
            u_internal=u_vec, specs=specs, base_cfg=base_cfg, simulator_binary=simulator_binary,
            runs=runs, temp_base=temp_base, backend=backend, keep_temp=keep_temp, retries=retries,
            objective_fn=objective_fn, objective_returns_features=True, default_features_fn=default_qd_features_unit
        )
        if isinstance(feats, (list, tuple, np.ndarray)):
            feats = tuple(map(float, np.asarray(feats, float).ravel()))
        else:
            feats = (float(feats),)
        return (float(f),), feats

    features_domain = ((0.0, 1.0), (0.0, 1.0))
    fitness_domain = ((-np.inf, np.inf),)

    grid = containers.Grid(
        shape=tuple(int(x) for x in qd_shape),
        max_items_per_bin=1,
        fitness_domain=fitness_domain,
        features_domain=features_domain
    )

    # Defaults (preserve current behavior) + override from YAML kwargs
    algo_hparams = {"mut_pb": 0.2, "eta": 20.0}
    if qd_algo_kwargs:
        algo_hparams.update(qd_algo_kwargs)

    algo = algorithms.RandomSearchMutPolyBounded(
        grid,
        budget=max_evals,
        batch_size=(qd_batch if qd_batch <= max_evals else max_evals),
        dimension=len(specs),
        optimisation_task="maximisation",
        ind_domain=(0., 1.),
        **algo_hparams,                     # <<< pass sel_pb, init_pb, mut_pb, eta, etc.
    )

    qdlogger = algorithms.TQDMAlgorithmLogger(algo, log_base_path=out_dir)
    with base.ParallelismManager("none") as pMgr:
        _ = algo.optimise(lambda ind: evaluate_one(np.clip(np.asarray(ind, float), 0.0, 1.0)),
                          executor=pMgr.executor, batch_mode=True)

    rows = []
    for elite in grid:
        if elite is None:
            continue
        f = getattr(elite, "fitness", None)
        fs = float(f[0]) if isinstance(f, (list, tuple)) else float(f.values[0])
        desc = getattr(elite, "features", ())
        desc = tuple(map(float, np.asarray(desc, float).ravel()))
        r = {"fitness": fs}
        for i, v in enumerate(desc):
            r[f"feat_{i}"] = v
        rows.append(r)

    print("\n" + algo.summary())
    plots.default_plots_grid(qdlogger, output_dir=out_dir)

    return {"qd_shape": list(qd_shape), "features_domain": features_domain, "budget": max_evals}




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
    qd_shape: Optional[str],
    qd_init_samples: int,
    qd_batch: int,
):
    os.makedirs(out_dir, exist_ok=True)

    with open(multi_config_path, "r", encoding="utf-8") as f:
        full_cfg = yaml.safe_load(f)
    specs = discover_optimization_domains(full_cfg)
    base_cfg = strip_optimization_domains(full_cfg)

    # Collect QD kwargs from YAML (either nested or flat)
    qd_algo_kwargs = {}
    opt_cfg = (full_cfg.get("optimization") or {})
    qd_cfg = (opt_cfg.get("qd") or {})
    if isinstance(qd_cfg.get("algo_kwargs"), dict):
        qd_algo_kwargs.update(qd_cfg["algo_kwargs"])
    for k in ("sel_pb", "init_pb", "mut_pb", "eta"):
        if k in qd_cfg:
            qd_algo_kwargs[k] = qd_cfg[k]

    objective_fn = load_objective(objective_path, objective_func_name)
    u0 = encode_x0_unit(specs)

    # History for single-solution optimizers
    history_rows = []
    best_f = -np.inf
    best_values: Optional[Dict[str, Any]] = None
    best_df: Optional[pd.DataFrame] = None

    def eval_one(u_internal: np.ndarray) -> float:
        nonlocal best_f, best_values, best_df
        f, feats, df, values = eval_fitness_and_features(
            u_internal=u_internal, specs=specs, base_cfg=base_cfg, simulator_binary=simulator_binary,
            runs=runs, temp_base=temp_base, backend=backend, keep_temp=keep_temp, retries=retries,
            objective_fn=objective_fn, objective_returns_features=False, default_features_fn=default_qd_features_maxstd_msd
        )
        now = time.time()
        nonlocal_eval_idx = len(history_rows) + 1
        if f >= best_f:
            best_f = f
            best_values = values
            best_df = df
        history_rows.append({
            "eval": nonlocal_eval_idx,
            "fitness": float(f),
            "best_so_far": float(best_f),
            "u_internal": json.dumps(list(map(float, u_internal))),
            "values": json.dumps(values),
            "timestamp": now,
        })
        logger.debug("Eval %d → fitness=%.6g | best=%.6g (u=%s)", nonlocal_eval_idx, f, best_f, np.array2string(u_internal, precision=3))
        return float(f)

    optimizer_name_l = optimizer_name.lower()
    if optimizer_name_l in ("mapelites"):
        # QD path
        shape_tuple: Tuple[int, ...]
        if qd_shape is None or qd_shape.strip() == "":
            #shape_tuple = (48, 48)  # sensible default for 2-D
            shape_tuple = (10, 10)  # sensible default for 2-D
        else:
            shape_tuple = tuple(int(s) for s in qd_shape.split(","))
        qd_info = run_qdpy_map_elites(
            specs=specs,
            base_cfg=base_cfg,
            simulator_binary=simulator_binary,
            runs=runs,
            temp_base=temp_base,
            backend=backend,
            keep_temp=keep_temp,
            retries=retries,
            objective_fn=objective_fn,
            default_features_fn=default_qd_features_maxstd_msd,
            qd_shape=shape_tuple,
            qd_init_samples=qd_init_samples,
            qd_batch=qd_batch,
            max_evals=max_evals,
            sigma0=sigma0,
            seed=seed,
            out_dir=out_dir,
            qd_algo_kwargs=qd_algo_kwargs,
        )
        # Persist a short summary JSON; return the QD artifacts instead of best config
        summary = {
            "optimizer": "qdpy-mapelites",
            "budget": max_evals,
            "qd": qd_info,
            #"files": {
            #    "qd_container_pkl": qd_info["qd_container_pkl"],
            #    "qd_archive_csv": qd_info["qd_archive_csv"],
            #    "qd_heatmap_png": os.path.join(out_dir, "qd_heatmap.png"),
            #},
        }
        with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        #logger.info("QD done. Container: %s | Archive CSV: %s", qd_info["qd_container_pkl"], qd_info["qd_archive_csv"])
        return

    # Single-solution optimization path (Random/CMA-ES)
    if optimizer_name_l == "random":
        opt = RandomSearch(dim=len(specs), max_evals=max_evals, seed=seed)
    elif optimizer_name_l == "cmaes":
        opt = CMAES(dim=len(specs), u0=u0, sigma0=sigma0, popsize=popsize, seed=seed, max_evals=max_evals)
    else:
        raise RuntimeError(f"Unknown optimizer: {optimizer_name}")

    logger.info("Starting optimization: %s | dim=%d | max_evals=%d (normalized space)", optimizer_name.upper(), len(specs), max_evals)
    opt.run(eval_one)

    # Save history
    hist_df = pd.DataFrame(history_rows)
    hist_csv = os.path.join(out_dir, "opt_history.csv")
    hist_df.to_csv(hist_csv, index=False)

    # Plot best-so-far
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

    if best_values is None or best_df is None:
        raise RuntimeError("No successful evaluations; cannot produce best config/results.")

    best_cfg = set_optimized_values_in_config(base_cfg, best_values)
    best_cfg = strip_optimization_domains(best_cfg)
    best_yaml = os.path.join(out_dir, "best_config.yaml")
    write_yaml(best_cfg, best_yaml)

    table = pa.Table.from_pandas(best_df)
    with open(best_yaml, "r", encoding="utf-8") as f:
        cfg_text = f.read()
    table = table.replace_schema_metadata({b"configuration": cfg_text.encode("utf-8")})
    best_feather = os.path.join(out_dir, "best_results.feather")
    paw.write_feather(table, best_feather)

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


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Optimize Pogosim controllers via CMA-ES/Random, or illuminate a repertoire via QDpy MAP-Elites."
    )
    p.add_argument("-c", "--config", required=True, help="Path to multi-value YAML.")
    p.add_argument("-S", "--simulator-binary", required=True, help="Path to pogosim executable.")
    p.add_argument("-r", "--runs", type=int, default=1, help="Number of runs per batch combination (default: 1).")
    p.add_argument("-t", "--temp-base", default="tmp_opt", help="Base temp directory (default: tmp_opt).")
    p.add_argument("-o", "--output-dir", default="opt_out", help="Directory to store outputs (default: opt_out).")
    p.add_argument("-O", "--objective", required=False, default=None,
                   help="Optional path to Python file exposing the fitness function. If omitted: DEFAULT fitness is mean MSD; QD features default to (polar order, straightness).")
    p.add_argument("--objective-func", default="compute_objective", help="Function name inside --objective (default: compute_objective).")

    p.add_argument("--optimizer", choices=["cmaes", "random", "mapelites"], default="cmaes",
                   help="Optimizer to use (default: cmaes).")

    p.add_argument("--max-evals", type=int, default=50, help="Max number of evaluations (budget).")
    p.add_argument("--sigma0", type=float, default=0.3, help="Initial step size in NORMALIZED space (default: 0.3).")
    p.add_argument("--popsize", type=int, default=None, help="CMA-ES population size override (optional).")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")

    p.add_argument("--backend", choices=["multiprocessing", "ray"], default="multiprocessing", help="Parallel backend for *batch evaluation* (default: multiprocessing).")
    p.add_argument("--keep-temp", action="store_true", help="Keep per-eval temp directories (debug).")
    p.add_argument("-R", "--retries", type=int, default=5, help="Relaunch a run upon simulator crash (default: 5).")
    p.add_argument("-v", "--verbose", action="store_true")

    # QD options
    p.add_argument("--qd-shape", default="48,48", help="Grid shape for MAP-Elites (comma-separated). Default: 48,48 (2-D).")
    p.add_argument("--qd-init-samples", type=int, default=64, help="Warm-start random samples to estimate feature bounds. Default: 64.")
    p.add_argument("--qd-batch", type=int, default=32, help="Batch of offspring per QD iteration (steady-state). Default: 32.")

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
            qd_shape=args.qd_shape,
            qd_init_samples=args.qd_init_samples,
            qd_batch=args.qd_batch,
        )
    except Exception as exc:
        logger.error("Fatal: %s", exc)
        traceback.print_exc()
        sys.exit(2)

if __name__ == "__main__":
    main()

# MODELINE "{{{1
# vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
# vim:foldmethod=marker
