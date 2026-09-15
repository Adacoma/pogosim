#!/usr/bin/env python3
"""Black-box and quality-diversity optimization for Pogosim controllers.

Float, integer, and categorical domains are represented in a normalized
``[0, 1]^d`` search space. Random Search and CMA-ES return a best configuration;
MAP-Elites uses QDpy to produce a repertoire. Objectives may return fitness or
``(fitness, features)``. The default fitness is mean MSD and the built-in QD
descriptors are polar order and trajectory straightness.

Each candidate is executed through Pogobatch's public local-campaign API, so
normal batch expansion, retries, task manifests, merging, and provenance are
shared between batch runs and optimization.
"""

from __future__ import annotations

import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import importlib.util
import json
import logging
import warnings
import math
import os
from pathlib import Path
import random
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from multiprocessing import current_process
from typing import Any, Callable, Dict, List, Optional, Tuple, Sequence
import traceback

logger = logging.getLogger("pogoptim")
_worker_logging_inited = False  # module-level flag

import matplotlib
matplotlib.use("Agg")  # Optimization is a headless batch workflow.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as paw
import yaml

if __package__:
    from .locomotion import compute_msd_per_agent
    from .pogobatch import LocalCampaignError, run_local_campaign
else:
    # Preserve direct execution from scripts/pogosim while installed/module use
    # follows normal package-relative imports.
    from locomotion import compute_msd_per_agent
    from pogobatch import LocalCampaignError, run_local_campaign

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
    required = ["robot_id", "time", "x", "y", "robot_category"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"Straightness requires columns: {', '.join(missing)}")
    group_columns = [
        column for column in ("run", "arena_file") if column in df.columns
    ] + ["robot_id"]
    g = df[group_columns + ["time", "x", "y", "robot_category"]].copy()
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
        s_vals = g.groupby(group_columns, sort=False).apply(
            one_track, include_groups=False
        )
    except TypeError:
        # Older pandas: explicitly select only the columns the function needs
        s_vals = (
            g.groupby(group_columns, sort=False)
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
    CTRL_KEYS = {"name", "default"}

    def rec(node: Any, dotted: str) -> None:
        if isinstance(node, dict):
            # explicit domain
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
                        raise RuntimeError(f"categorical domain {dotted} needs a non-empty 'choices' list (or 'batch_options').")
                    init_val = dom.get("init", node.get("default_option"))
                    init_idx = choices.index(init_val) if (init_val in choices) else 0
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

            # NEW: hierarchical domain (treat as categorical over branch names)
            if "batch_hierarchical_options" in node and isinstance(node["batch_hierarchical_options"], dict):
                mapping = node["batch_hierarchical_options"]
                names = [k for k in mapping.keys() if k not in CTRL_KEYS]
                if names:
                    init_val = node.get("default_option")
                    init_idx = names.index(init_val) if (init_val in names) else 0
                    specs.append(VarSpec(path=dotted, kind="categorical", choices=names, init=float(init_idx)))

            # Recurse
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


def _set_dotted_value(cfg: Dict[str, Any], dotted: str, value: Any) -> None:
    """Set a dotted mapping path, creating intermediate mappings as needed."""
    parts = dotted.split(".") if dotted else []
    if not parts:
        raise KeyError("Cannot assign an empty configuration path")
    node: Dict[str, Any] = cfg
    for part in parts[:-1]:
        child = node.get(part)
        if child is None:
            child = {}
            node[part] = child
        if not isinstance(child, dict):
            raise KeyError(f"Cannot create {dotted}: {part} is not a mapping")
        node = child
    node[parts[-1]] = value


def set_optimized_values_in_config(base_cfg: Dict[str, Any], values: Dict[str, Any]) -> Dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    for dotted, val in values.items():
        try:
            node = _resolve_node(cfg, dotted)
            if isinstance(node, dict):
                # If this node holds a hierarchical mapping, shrink it to the chosen branch.
                if "batch_hierarchical_options" in node and isinstance(node["batch_hierarchical_options"], dict):
                    mapping = node["batch_hierarchical_options"]
                    if val not in mapping:
                        raise KeyError(f"Chosen hierarchical alternative '{val}' not found at {dotted}")
                    chosen = mapping[val]
                    # Keep alias if any, plus only the chosen branch
                    alias = mapping.get("name")
                    new_map = {"batch_hierarchical_options": {val: copy.deepcopy(chosen)}}
                    if alias is not None:
                        new_map["batch_hierarchical_options"]["name"] = alias
                    # Replace mapping in place
                    node["batch_hierarchical_options"] = new_map["batch_hierarchical_options"]
                    # Optionally remember for reproducibility
                    node["default_option"] = val
                else:
                    # regular dict target (incl. list-based batch_options owners)
                    node["default_option"] = val
            else:
                # scalar path
                _set_dotted_value(cfg, dotted, val)
        except KeyError:
            # Create a dict owner with default_option when the path is missing
            _set_dotted_value(cfg, dotted, {"default_option": val})
    return cfg


def write_yaml(obj: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


# ----------------------------------------------------------------------------
# Run one candidate (batch grid x seeds) and compute fitness/features
# ----------------------------------------------------------------------------

PENALTY = -1e9
PORTABLE_SEED_LIMIT = 2**31


class ObjectiveContractError(RuntimeError):
    """The objective's return value is incompatible with the selected optimizer."""


@dataclass
class EvaluationResult:
    index: int
    u_internal: np.ndarray
    values: Dict[str, Any]
    base_seeds: tuple[int, ...]
    effective_seeds: tuple[int, ...]
    fitness: float = PENALTY
    features: np.ndarray | None = None
    dataframe: pd.DataFrame | None = None
    success: bool = False
    error: str | None = None


class CandidateSeedAllocator:
    """Allocate deterministic, non-overlapping initial and retry seed blocks."""

    def __init__(
        self,
        *,
        optimizer_seed: int,
        evaluations: int,
        runs: int,
        retries: int,
        retry_new_seed: bool,
    ) -> None:
        attempts = retries + 1 if retry_new_seed else 1
        self.block_size = runs * attempts
        total_slots = evaluations * self.block_size
        if total_slots <= 0 or total_slots > PORTABLE_SEED_LIMIT:
            raise RuntimeError("Requested evaluation/run/retry budget exceeds the portable seed space")
        seed_rng = np.random.default_rng(
            np.random.SeedSequence([optimizer_seed, 0x504F474F])
        )
        self.offset = int(seed_rng.integers(0, PORTABLE_SEED_LIMIT - total_slots + 1))
        self.runs = runs

    def seeds_for(self, evaluation_index: int) -> tuple[int, ...]:
        start = self.offset + evaluation_index * self.block_size
        return tuple(range(start, start + self.runs))


def _effective_seeds(df: pd.DataFrame, base_seeds: tuple[int, ...]) -> tuple[int, ...]:
    if "seed" not in df.columns:
        return base_seeds
    if "run" in df.columns:
        rows = df[["run", "seed"]].drop_duplicates().sort_values("run")
        return tuple(int(seed) for seed in rows["seed"].tolist())
    return tuple(int(seed) for seed in df["seed"].drop_duplicates().tolist())


def run_evaluation(
    cfg_for_eval: Dict[str, Any],
    simulator_binary: str,
    seeds: tuple[int, ...],
    temp_base: str,
    backend: str,
    batch_jobs: int,
    keep_temp: bool,
    retries: int,
    retry_new_seed: bool,
) -> pd.DataFrame:
    Path(temp_base).mkdir(parents=True, exist_ok=True)
    eval_tmp = Path(tempfile.mkdtemp(prefix="eval_", dir=temp_base))
    completed = False
    try:
        cfg_for_eval = copy.deepcopy(cfg_for_eval)
        batch_settings = cfg_for_eval.setdefault("pogobatch", {})
        if not isinstance(batch_settings, dict):
            raise RuntimeError("The top-level 'pogobatch' setting must be a mapping")
        rnc: List[str] = list(batch_settings.get("result_new_columns", []) or [])
        arena_paths = _find_dotted_paths_for_key(cfg_for_eval, "arena_file")
        arena_paths = [path for path in arena_paths if not path.startswith("pogobatch.")]
        arena_path = min(arena_paths, key=len) if arena_paths else None
        if arena_path and arena_path not in rnc:
            rnc.append(arena_path)
        if rnc:
            batch_settings["result_new_columns"] = rnc

        cfg_path = eval_tmp / "multi.yaml"
        write_yaml(cfg_for_eval, str(cfg_path))
        result = run_local_campaign(
            cfg_path,
            simulator_binary,
            seeds=seeds,
            temp_base=eval_tmp / "tmp",
            output_dir=eval_tmp / "out",
            backend=backend,
            jobs=batch_jobs,
            keep_temp=keep_temp,
            retries=retries,
            retry_new_seed=retry_new_seed,
            simulator_output="quiet",
        )
        if not result.outputs:
            raise RuntimeError("No output files produced by Pogobatch")
        frames = [pd.read_feather(path) for path in result.outputs if path.exists()]
        if not frames:
            raise RuntimeError("Pogobatch outputs are missing or unreadable")
        df = pd.concat(frames, ignore_index=True)
        for column in list(df.columns):
            if column.endswith(".arena_file") and "arena_file" not in df.columns:
                df = df.rename(columns={column: "arena_file"})
            elif column.endswith(".arena_file") and "arena_file" in df.columns:
                df = df.drop(columns=[column])
        completed = True
        return df
    finally:
        # Failed Pogobatch campaigns deliberately retain their diagnostic shards.
        if completed and not keep_temp:
            shutil.rmtree(eval_tmp, ignore_errors=True)


def evaluate_candidate(
    *,
    index: int,
    u_internal: np.ndarray,
    specs: List[VarSpec],
    base_cfg: Dict[str, Any],
    simulator_binary: str,
    seeds: tuple[int, ...],
    temp_base: str,
    backend: str,
    batch_jobs: int,
    keep_temp: bool,
    retries: int,
    retry_new_seed: bool,
    objective_fn: Callable[[pd.DataFrame], Any],
    default_features_fn: Callable[[pd.DataFrame], np.ndarray],
    qd_mode: bool,
    custom_objective: bool,
    qd_domains_explicit: bool,
    feature_domains: tuple[tuple[float, float], ...],
) -> EvaluationResult:
    values = decode_unit_vector(specs, u_internal)
    result = EvaluationResult(
        index=index,
        u_internal=np.asarray(u_internal, dtype=float),
        values=values,
        base_seeds=seeds,
        effective_seeds=seeds,
    )
    try:
        df = run_evaluation(
            set_optimized_values_in_config(base_cfg, values),
            simulator_binary,
            seeds,
            temp_base,
            backend,
            batch_jobs,
            keep_temp,
            retries,
            retry_new_seed,
        )
    except LocalCampaignError as exc:
        result.error = f"{exc}; campaign_dir={exc.campaign_dir}"
        logger.error("Evaluation %d failed; penalizing candidate: %s", index + 1, exc)
        return result

    result.dataframe = df
    result.effective_seeds = _effective_seeds(df, seeds)
    try:
        objective_output = objective_fn(df)
    except Exception as exc:  # An objective can fail for candidate-specific data.
        result.error = f"objective failed: {exc}"
        logger.error("Evaluation %d objective failed; penalizing candidate: %s", index + 1, exc)
        return result

    custom_features = False
    if isinstance(objective_output, (tuple, list)) and len(objective_output) >= 2:
        fitness = float(objective_output[0])
        raw_features = objective_output[1]
        if isinstance(raw_features, dict):
            raw_features = list(raw_features.values())
        features = np.asarray(raw_features, dtype=float).ravel()
        custom_features = True
    else:
        fitness = float(objective_output)
        features = np.asarray(default_features_fn(df), dtype=float).ravel()

    if qd_mode:
        if custom_objective and custom_features and not qd_domains_explicit:
            raise ObjectiveContractError(
                "A custom MAP-Elites descriptor requires optimization.qd.features_domain"
            )
        if len(features) != len(feature_domains):
            raise ObjectiveContractError(
                f"Objective returned {len(features)} descriptors; expected {len(feature_domains)}"
            )
        if not np.all(np.isfinite(features)):
            result.error = "objective returned non-finite descriptors"
            return result
        if any(not (lo <= value <= hi) for value, (lo, hi) in zip(features, feature_domains)):
            result.error = "objective descriptors lie outside optimization.qd.features_domain"
            return result

    if not np.isfinite(fitness):
        result.error = "objective returned non-finite fitness"
        return result
    result.fitness = float(fitness)
    result.features = features
    result.success = True
    return result



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
    def __init__(self, dim: int, max_evals: int, seed: int = 42, batch_size: int = 1):
        super().__init__(dim)
        self.max_evals = max_evals
        self.batch_size = max(1, batch_size)
        self.rng = np.random.default_rng(seed)

    def run(self, evaluate_many):
        evals = 0
        while evals < self.max_evals:
            count = min(self.batch_size, self.max_evals - evals)
            candidates = [
                self.rng.uniform(0.0, 1.0, size=self.dim).astype(float)
                for _ in range(count)
            ]
            for candidate, fitness in zip(candidates, evaluate_many(candidates)):
                if fitness > self.best_f:
                    self.best_f = fitness
                    self.best_u = candidate.copy()
            evals += count
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

    def run(self, evaluate_many):
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
            for u_eval, f in zip(xs_eval, evaluate_many(xs_eval)):
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
    evaluate_many: Callable[[Sequence[np.ndarray]], List[EvaluationResult]],
    qd_shape: Sequence[int],
    feature_domains: tuple[tuple[float, float], ...],
    qd_batch: int,
    max_evals: int,
    seed: int,
    out_dir: str,
    qd_algo_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    try:
        from qdpy import algorithms, containers, plots
    except ImportError as exc:
        raise RuntimeError(
            "MAP-Elites requested but QDpy is unavailable; install pogosim[optim]"
        ) from exc

    # QDpy uses the module-level Python and NumPy RNGs when producing genomes.
    library_seed = int(np.random.SeedSequence(seed).generate_state(1)[0])
    random.seed(library_seed)
    np.random.seed(library_seed)
    max_evals = int(max_evals)

    grid = containers.Grid(
        shape=tuple(int(x) for x in qd_shape),
        max_items_per_bin=1,
        fitness_domain=((-np.inf, np.inf),),
        features_domain=feature_domains,
    )

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
        **algo_hparams,
    )

    qdlogger = algorithms.AlgorithmLogger(algo, log_base_path=out_dir, verbose=False)
    successful_genomes: set[tuple[float, ...]] = set()
    evaluations = 0
    while evaluations < max_evals:
        count = min(qd_batch, max_evals - evaluations)
        individuals = [algo.ask() for _ in range(count)]
        candidates = [
            np.clip(np.asarray(individual, dtype=float), 0.0, 1.0)
            for individual in individuals
        ]
        results = evaluate_many(candidates)
        for individual, result in zip(individuals, results):
            if result.success and result.features is not None:
                features = tuple(float(value) for value in result.features)
                successful_genomes.add(tuple(float(value) for value in individual))
            else:
                # A failed point remains a worst-fitness parent but is never
                # exported as an elite; this lets MAP-Elites keep exploring.
                features = tuple((lo + hi) / 2.0 for lo, hi in feature_domains)
            algo.tell(individual, fitness=(result.fitness,), features=features)
        evaluations += count

    rows = []
    for elite in grid:
        if elite is None or tuple(float(value) for value in elite) not in successful_genomes:
            continue
        fitness = getattr(elite, "fitness", None)
        score = (
            float(fitness[0])
            if isinstance(fitness, (list, tuple))
            else float(fitness.values[0])
        )
        desc = getattr(elite, "features", ())
        desc = tuple(map(float, np.asarray(desc, float).ravel()))
        genome = np.asarray(elite, dtype=float)
        row: Dict[str, Any] = {"fitness": score}
        row.update({f"feature_{i}": value for i, value in enumerate(desc)})
        row.update({f"u_{i}": float(value) for i, value in enumerate(genome)})
        row["values"] = json.dumps(decode_unit_vector(specs, genome), sort_keys=True)
        rows.append(row)

    output_path = Path(out_dir)
    archive_path = output_path / "qd_archive.csv"
    pd.DataFrame(rows).to_csv(archive_path, index=False)
    pickle_path = output_path / "qd_final.p"
    qdlogger.save(str(pickle_path))
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=r"set_ticklabels\(\) should only be used"
            )
            plots.default_plots_grid(qdlogger, output_dir=str(output_path))
    except Exception as exc:
        logger.warning("Could not generate QDpy grid plots: %s", exc)

    logger.debug("%s", algo.summary())
    logger.info("MAP-Elites completed with %d exported elite(s)", len(rows))
    return {
        "qd_shape": list(qd_shape),
        "features_domain": [list(domain) for domain in feature_domains],
        "budget": max_evals,
        "elite_count": len(rows),
        "files": {
            "container_pickle": str(pickle_path),
            "archive_csv": str(archive_path),
        },
    }




# ----------------------------------------------------------------------------
# Main optimization driver
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class OptimizationSettings:
    algorithm: str = "cmaes"
    seed: int = 42
    budget: int = 50
    runs: int = 1
    retries: int = 5
    retry_new_seed: bool = True
    backend: str = "multiprocessing"
    candidate_jobs: int = 1
    batch_jobs: int = 0
    output_dir: str = "opt_out"
    temp_base: str = "tmp_opt"
    keep_temp: bool = False
    objective_path: str | None = None
    objective_func: str = "compute_objective"
    sigma0: float = 0.3
    popsize: int | None = None
    qd_shape: tuple[int, ...] = (48, 48)
    qd_batch: int = 32
    qd_features_domain: tuple[tuple[float, float], ...] = ((0.0, 1.0), (0.0, 1.0))
    qd_domains_explicit: bool = False
    qd_algo_kwargs: Dict[str, Any] | None = None


def _parse_shape(value: Any) -> tuple[int, ...]:
    if isinstance(value, str):
        values = value.split(",")
    elif isinstance(value, (list, tuple)):
        values = value
    else:
        raise RuntimeError("optimization.qd.shape must be a list or comma-separated string")
    shape = tuple(int(item) for item in values)
    if not shape or any(item <= 0 for item in shape):
        raise RuntimeError("Every optimization.qd.shape entry must be positive")
    return shape


def _parse_feature_domains(value: Any) -> tuple[tuple[float, float], ...]:
    if isinstance(value, str):
        value = [item.split(":") for item in value.split(",")]
    if not isinstance(value, (list, tuple)):
        raise RuntimeError("optimization.qd.features_domain must be a sequence")
    domains: list[tuple[float, float]] = []
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise RuntimeError("Each QD feature domain must contain [minimum, maximum]")
        lo, hi = float(item[0]), float(item[1])
        if not math.isfinite(lo) or not math.isfinite(hi) or lo >= hi:
            raise RuntimeError("QD feature domains require finite minimum < maximum")
        domains.append((lo, hi))
    return tuple(domains)


def resolve_optimization_settings(
    full_cfg: Dict[str, Any], cli: argparse.Namespace
) -> OptimizationSettings:
    raw = full_cfg.get("optimization", {}) or {}
    if not isinstance(raw, dict):
        raise RuntimeError("The top-level 'optimization' setting must be a mapping")
    values: Dict[str, Any] = {
        field: getattr(OptimizationSettings(), field)
        for field in OptimizationSettings.__dataclass_fields__
    }
    for key in (
        "algorithm", "seed", "budget", "runs", "retries", "retry_new_seed",
        "backend", "output_dir", "temp_base", "keep_temp",
    ):
        if key in raw:
            values[key] = raw[key]

    parallel = raw.get("parallelism", {}) or {}
    if not isinstance(parallel, dict):
        raise RuntimeError("optimization.parallelism must be a mapping")
    for key in ("candidate_jobs", "batch_jobs"):
        if key in parallel:
            values[key] = parallel[key]

    objective = raw.get("objective", {}) or {}
    if not isinstance(objective, dict):
        raise RuntimeError("optimization.objective must be a mapping")
    values["objective_path"] = objective.get("path", values["objective_path"])
    values["objective_func"] = objective.get("func", values["objective_func"])

    cmaes = raw.get("cmaes", {}) or {}
    if not isinstance(cmaes, dict):
        raise RuntimeError("optimization.cmaes must be a mapping")
    values["sigma0"] = cmaes.get("sigma0", values["sigma0"])
    values["popsize"] = cmaes.get("popsize", values["popsize"])

    qd = raw.get("qd", {}) or {}
    if not isinstance(qd, dict):
        raise RuntimeError("optimization.qd must be a mapping")
    values["qd_shape"] = _parse_shape(qd.get("shape", values["qd_shape"]))
    values["qd_batch"] = qd.get("batch", values["qd_batch"])
    if "features_domain" in qd:
        values["qd_features_domain"] = _parse_feature_domains(qd["features_domain"])
        values["qd_domains_explicit"] = True
    algo_kwargs = dict(qd.get("algo_kwargs", {}) or {})
    for key in ("sel_pb", "init_pb", "mut_pb", "eta"):
        if key in qd:
            algo_kwargs[key] = qd[key]
    values["qd_algo_kwargs"] = algo_kwargs
    if "init_samples" in qd or hasattr(cli, "qd_init_samples"):
        logger.warning(
            "qd.init_samples/--qd-init-samples is deprecated and ignored; "
            "configure qd.features_domain"
        )

    cli_mapping = {
        "optimizer": "algorithm",
        "max_evals": "budget",
        "objective": "objective_path",
        "objective_func": "objective_func",
        "qd_shape": "qd_shape",
        "qd_batch": "qd_batch",
        "qd_features_domain": "qd_features_domain",
    }
    for cli_key, target in cli_mapping.items():
        if hasattr(cli, cli_key):
            value = getattr(cli, cli_key)
            if target == "qd_shape":
                value = _parse_shape(value)
            elif target == "qd_features_domain":
                value = _parse_feature_domains(value)
                values["qd_domains_explicit"] = True
            values[target] = value
    for key in (
        "runs", "temp_base", "output_dir", "sigma0", "popsize", "seed",
        "backend", "keep_temp", "retries", "retry_new_seed",
        "candidate_jobs", "batch_jobs",
    ):
        if hasattr(cli, key):
            values[key] = getattr(cli, key)

    settings = OptimizationSettings(**values)
    if settings.algorithm not in {"random", "cmaes", "mapelites"}:
        raise RuntimeError(f"Unknown optimizer: {settings.algorithm}")
    if settings.backend not in {"sequential", "multiprocessing", "ray"}:
        raise RuntimeError(f"Unknown batch backend: {settings.backend}")
    if settings.seed < 0 or settings.budget <= 0 or settings.runs <= 0:
        raise RuntimeError("seed must be non-negative and budget/runs must be positive")
    if settings.retries < 0 or settings.candidate_jobs <= 0 or settings.batch_jobs < 0:
        raise RuntimeError("retries/batch_jobs must be non-negative and candidate_jobs positive")
    if settings.sigma0 <= 0 or (settings.popsize is not None and settings.popsize <= 0):
        raise RuntimeError("CMA-ES sigma0 and popsize must be positive")
    if settings.qd_batch <= 0:
        raise RuntimeError("optimization.qd.batch must be positive")
    if len(settings.qd_shape) != len(settings.qd_features_domain):
        raise RuntimeError("QD shape and features_domain must have the same dimension")
    return settings


def _available_cpu_count() -> int:
    process_count = getattr(os, "process_cpu_count", None)
    count = process_count() if process_count is not None else os.cpu_count()
    return max(1, count or 1)


@contextmanager
def _shared_ray_runtime(backend: str, candidate_jobs: int):
    if backend != "ray" or candidate_jobs <= 1:
        yield
        return
    try:
        import ray
    except ImportError as exc:
        raise RuntimeError("Ray backend requested but Ray is not installed") from exc
    owns_runtime = not ray.is_initialized()
    if owns_runtime:
        ray.init(ignore_reinit_error=True)
    try:
        yield
    finally:
        if owns_runtime:
            ray.shutdown()


def optimize(
    multi_config_path: str,
    simulator_binary: str,
    settings: OptimizationSettings,
) -> Dict[str, Any]:
    config_path = Path(multi_config_path)
    with config_path.open("r", encoding="utf-8") as stream:
        full_cfg = yaml.safe_load(stream)
    if not isinstance(full_cfg, dict):
        raise RuntimeError("Optimization configuration must be a YAML mapping")
    if os.sep in simulator_binary and not Path(simulator_binary).is_file():
        raise RuntimeError(f"Simulator binary does not exist: {simulator_binary}")
    if os.sep not in simulator_binary and shutil.which(simulator_binary) is None:
        raise RuntimeError(f"Simulator binary is not on PATH: {simulator_binary}")

    specs = discover_optimization_domains(full_cfg)
    scientific_cfg = copy.deepcopy(full_cfg)
    scientific_cfg.pop("optimization", None)
    base_cfg = strip_optimization_domains(scientific_cfg)
    objective_fn = load_objective(settings.objective_path, settings.objective_func)
    output_dir = Path(settings.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    Path(settings.temp_base).mkdir(parents=True, exist_ok=True)

    cpu_count = _available_cpu_count()
    candidate_jobs = min(settings.candidate_jobs, settings.budget, cpu_count)
    requested_batch_jobs = settings.batch_jobs or cpu_count
    batch_jobs = min(requested_batch_jobs, max(1, cpu_count // candidate_jobs))
    if settings.backend == "sequential":
        batch_jobs = 1
    if candidate_jobs != settings.candidate_jobs or (
        settings.batch_jobs > 0 and batch_jobs != settings.batch_jobs
    ):
        logger.warning(
            "Capping parallelism to candidate_jobs=%d, batch_jobs=%d for %d CPUs",
            candidate_jobs,
            batch_jobs,
            cpu_count,
        )
    else:
        logger.info("Parallelism: candidate_jobs=%d, batch_jobs=%d", candidate_jobs, batch_jobs)

    seed_allocator = CandidateSeedAllocator(
        optimizer_seed=settings.seed,
        evaluations=settings.budget,
        runs=settings.runs,
        retries=settings.retries,
        retry_new_seed=settings.retry_new_seed,
    )
    history_rows: List[Dict[str, Any]] = []
    next_evaluation = 0
    best_fitness = -np.inf
    best_values: Dict[str, Any] | None = None
    best_df: pd.DataFrame | None = None

    def evaluate_many(candidates: Sequence[np.ndarray]) -> List[EvaluationResult]:
        nonlocal next_evaluation, best_fitness, best_values, best_df
        indexed = [
            (next_evaluation + offset, np.asarray(candidate, dtype=float))
            for offset, candidate in enumerate(candidates)
        ]
        next_evaluation += len(indexed)

        def evaluate(item: tuple[int, np.ndarray]) -> EvaluationResult:
            index, candidate = item
            return evaluate_candidate(
                index=index,
                u_internal=candidate,
                specs=specs,
                base_cfg=base_cfg,
                simulator_binary=simulator_binary,
                seeds=seed_allocator.seeds_for(index),
                temp_base=settings.temp_base,
                backend=settings.backend,
                batch_jobs=batch_jobs,
                keep_temp=settings.keep_temp,
                retries=settings.retries,
                retry_new_seed=settings.retry_new_seed,
                objective_fn=objective_fn,
                default_features_fn=default_qd_features_unit,
                qd_mode=settings.algorithm == "mapelites",
                custom_objective=settings.objective_path is not None,
                qd_domains_explicit=settings.qd_domains_explicit,
                feature_domains=settings.qd_features_domain,
            )

        workers = min(candidate_jobs, len(indexed))
        if workers <= 1:
            results = [evaluate(item) for item in indexed]
        else:
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="pogoptim") as executor:
                results = list(executor.map(evaluate, indexed))

        for result in results:
            if result.success and result.fitness >= best_fitness:
                best_fitness = result.fitness
                best_values = result.values
                best_df = result.dataframe
            retry_attempts = None
            if result.dataframe is not None and "retry_attempt" in result.dataframe.columns:
                retry_attempts = int(result.dataframe["retry_attempt"].max())
            history_rows.append({
                "eval": result.index + 1,
                "status": "success" if result.success else "failed",
                "fitness": result.fitness,
                "best_so_far": best_fitness if np.isfinite(best_fitness) else None,
                "features": json.dumps(result.features.tolist()) if result.features is not None else None,
                "u_internal": json.dumps(result.u_internal.tolist()),
                "values": json.dumps(result.values, sort_keys=True),
                "base_seeds": json.dumps(result.base_seeds),
                "effective_seeds": json.dumps(result.effective_seeds),
                "max_retry_attempt": retry_attempts,
                "error": result.error,
                "timestamp": time.time(),
            })
        return results

    qd_info: Dict[str, Any] | None = None
    logger.info(
        "Starting optimization: %s | dim=%d | max_evals=%d",
        settings.algorithm.upper(),
        len(specs),
        settings.budget,
    )
    with _shared_ray_runtime(settings.backend, candidate_jobs):
        if settings.algorithm == "random":
            optimizer = RandomSearch(
                len(specs), settings.budget, settings.seed, batch_size=candidate_jobs
            )
            optimizer.run(lambda points: [result.fitness for result in evaluate_many(points)])
        elif settings.algorithm == "cmaes":
            library_seed = (
                int(np.random.SeedSequence(settings.seed).generate_state(1)[0])
                % (2**31 - 1)
            ) + 1
            optimizer = CMAES(
                len(specs), encode_x0_unit(specs), settings.sigma0,
                settings.popsize, library_seed, settings.budget,
            )
            optimizer.run(lambda points: [result.fitness for result in evaluate_many(points)])
        else:
            qd_info = run_qdpy_map_elites(
                specs,
                evaluate_many,
                settings.qd_shape,
                settings.qd_features_domain,
                settings.qd_batch,
                settings.budget,
                settings.seed,
                str(output_dir),
                settings.qd_algo_kwargs,
            )

    history = pd.DataFrame(history_rows).sort_values("eval")
    history_path = output_dir / "opt_history.csv"
    history.to_csv(history_path, index=False)
    success_count = int((history["status"] == "success").sum())
    common_summary: Dict[str, Any] = {
        "status": "complete" if success_count else "failed",
        "optimizer": settings.algorithm,
        "budget": settings.budget,
        "successful_evaluations": success_count,
        "failed_evaluations": len(history) - success_count,
        "seed": settings.seed,
        "seed_block_offset": seed_allocator.offset,
        "runs": settings.runs,
        "parallelism": {
            "candidate_jobs": candidate_jobs,
            "batch_jobs": batch_jobs,
            "backend": settings.backend,
        },
        "settings": {
            "retries": settings.retries,
            "retry_new_seed": settings.retry_new_seed,
            "keep_temp": settings.keep_temp,
            "objective_path": settings.objective_path,
            "objective_func": settings.objective_func,
            "cmaes": {"sigma0": settings.sigma0, "popsize": settings.popsize},
            "qd": {
                "shape": list(settings.qd_shape),
                "batch": settings.qd_batch,
                "features_domain": [list(domain) for domain in settings.qd_features_domain],
                "algo_kwargs": settings.qd_algo_kwargs,
            },
        },
        "files": {"history_csv": str(history_path)},
    }

    summary_path = output_dir / "summary.json"
    if not success_count:
        summary_path.write_text(json.dumps(common_summary, indent=2), encoding="utf-8")
        raise RuntimeError("No successful evaluations; see opt_history.csv for failures")

    if settings.algorithm == "mapelites":
        common_summary["qd"] = qd_info
        common_summary["files"].update((qd_info or {}).get("files", {}))
        summary_path.write_text(json.dumps(common_summary, indent=2), encoding="utf-8")
        return common_summary

    plot_path = output_dir / "fitness_vs_eval.png"
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(history["eval"], history["best_so_far"], label="best so far")
    axis.set(xlabel="evaluation", ylabel="fitness")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(plot_path, dpi=144)
    plt.close(figure)

    assert best_values is not None and best_df is not None
    best_config = strip_optimization_domains(
        set_optimized_values_in_config(base_cfg, best_values)
    )
    best_config_path = output_dir / "best_config.yaml"
    write_yaml(best_config, str(best_config_path))
    table = pa.Table.from_pandas(best_df)
    metadata = dict(table.schema.metadata or {})
    metadata[b"configuration"] = best_config_path.read_bytes()
    table = table.replace_schema_metadata(metadata)
    best_results_path = output_dir / "best_results.feather"
    paw.write_feather(table, best_results_path)

    common_summary["best_fitness"] = best_fitness
    common_summary["best_values"] = best_values
    common_summary["files"].update({
        "plot_png": str(plot_path),
        "best_config_yaml": str(best_config_path),
        "best_results_feather": str(best_results_path),
    })
    summary_path.write_text(json.dumps(common_summary, indent=2), encoding="utf-8")
    logger.info("Done. Best fitness: %.6g", best_fitness)
    return common_summary


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Optimize Pogosim controllers with Random Search, CMA-ES, or MAP-Elites.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-c", "--config", required=True, help="Optimization-enabled Pogosim YAML")
    parser.add_argument("-S", "--simulator-binary", required=True, help="Pogosim executable")
    parser.add_argument("-v", "--verbose", action="store_true")

    # SUPPRESS distinguishes an omitted option from an explicit YAML override.
    optional = {"default": argparse.SUPPRESS}
    parser.add_argument("-r", "--runs", type=int, **optional)
    parser.add_argument("-t", "--temp-base", **optional)
    parser.add_argument("-o", "--output-dir", **optional)
    parser.add_argument("-O", "--objective", **optional)
    parser.add_argument("--objective-func", **optional)
    parser.add_argument("--optimizer", choices=("cmaes", "random", "mapelites"), **optional)
    parser.add_argument("--max-evals", type=int, **optional)
    parser.add_argument("--sigma0", type=float, **optional)
    parser.add_argument("--popsize", type=int, **optional)
    parser.add_argument("--seed", type=int, **optional)
    parser.add_argument(
        "--backend", choices=("multiprocessing", "ray", "sequential"), **optional
    )
    parser.add_argument("--candidate-jobs", type=int, **optional)
    parser.add_argument("--batch-jobs", type=int, **optional)
    parser.add_argument(
        "--keep-temp", action=argparse.BooleanOptionalAction, **optional
    )
    parser.add_argument("-R", "--retries", type=int, **optional)
    retry_group = parser.add_mutually_exclusive_group()
    retry_group.add_argument(
        "--retry-new-seed", dest="retry_new_seed", action="store_true", **optional
    )
    retry_group.add_argument(
        "--retry-same-seed", dest="retry_new_seed", action="store_false", **optional
    )
    parser.add_argument("--qd-shape", **optional)
    parser.add_argument("--qd-batch", type=int, **optional)
    parser.add_argument(
        "--qd-features-domain",
        metavar="MIN:MAX,MIN:MAX",
        help="Explicit MAP-Elites descriptor ranges",
        **optional,
    )
    parser.add_argument(
        "--qd-init-samples",
        type=int,
        help="Deprecated compatibility option; use --qd-features-domain",
        **optional,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    init_logging(args.verbose)
    batch_logger = logging.getLogger("pogobatch")
    batch_logger.handlers.clear()
    for handler in logger.handlers:
        batch_logger.addHandler(handler)
    batch_logger.propagate = False
    batch_logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)

    try:
        with open(args.config, "r", encoding="utf-8") as stream:
            full_cfg = yaml.safe_load(stream)
        if not isinstance(full_cfg, dict):
            raise RuntimeError("Optimization configuration must be a YAML mapping")
        settings = resolve_optimization_settings(full_cfg, args)
        optimize(args.config, args.simulator_binary, settings)
        return 0
    except Exception as exc:
        logger.error("Fatal: %s", exc)
        if args.verbose:
            traceback.print_exc()
        return 2

if __name__ == "__main__":
    raise SystemExit(main())

# MODELINE "{{{1
# vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
# vim:foldmethod=marker
