#!/usr/bin/env python3
"""Pogosim batch runner with local and Rundra-backed execution.

The CLI is intentionally split into six operations:

  pogobatch help     Show general help or help for a specific subcommand.
  pogobatch plan     Expand a Pogosim batch configuration without executing it.
  pogobatch task     Execute one configuration, resolving batch defaults when needed.
  pogobatch run      Execute a complete batch locally, then merge task shards.
  pogobatch merge    Merge self-describing task shards into result_*.feather files.
  pogobatch cluster  Run once through Rundra, materialize outputs, merge, and clean up.

`task` is the atomic execution primitive.  It writes a small
`pogobatch_task.json` sidecar after a successful simulation.  `run` and `cluster`
therefore share exactly the same merge path, and `merge` is independent of the
executor that produced the task directories.

Batch-only settings live under a reserved top-level ``pogobatch:`` mapping.
The canonical result settings are ``pogobatch.result_filename_format`` and
``pogobatch.result_new_columns``; Rundra source exclusions are configured with
``pogobatch.rundra.sync.exclude``.  This block is stripped before Pogosim sees
the effective YAML.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import logging
import numbers
import os
from pathlib import Path
import re
import secrets
import shlex
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from multiprocessing import Pool
from typing import Any, Sequence
import uuid

import pyarrow as pa
import pyarrow.feather as feather
import yaml

from pogosim import __version__
from pogosim import utils


logger = logging.getLogger("pogobatch")
logger.addHandler(logging.NullHandler())

TASK_MANIFEST_NAME = "pogobatch_task.json"
TASK_MANIFEST_VERSION = 2
SUPPORTED_TASK_MANIFEST_VERSIONS = frozenset({1, 2})
RUN_MANIFEST_NAME = "pogobatch_run.json"
RUNDRA_WRAPPER_KEY = "_pogobatch_task"
RUNDRA_METADATA_KEY = "_rundr"
POGOBATCH_CONFIG_KEY = "pogobatch"
POGOBATCH_CHOICE_KEY = "_pogobatch_choice"
LEGACY_RESULT_KEYS = ("result_filename_format", "result_new_columns")
POGOBATCH_SCRIPT_VERSION = "21"


class PogobatchError(RuntimeError):
    """Expected user-facing Pogobatch failure."""


class RundraOperationError(PogobatchError):
    """Structured failure returned by Rundra's JSON interface."""

    def __init__(
        self,
        code: str,
        message: str,
        details: Any | None = None,
        *,
        operation: str | None = None,
    ) -> None:
        self.code = code
        self.rundra_message = message
        self.details = details
        self.operation = operation
        suffix = f"; details={details}" if details else ""
        super().__init__(f"Rundra [{code}]: {message}{suffix}")


class DotDict(dict):
    """Allow attribute-style access while preserving normal mapping behavior."""

    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


@dataclass(frozen=True)
class Combination:
    ordinal: int
    config: dict[str, Any]
    result_filename: str
    extra_columns: dict[str, Any]
    config_hash: str


@dataclass(frozen=True)
class LocalTaskSpec:
    combination_ordinal: int
    config: dict[str, Any]
    result_filename: str
    extra_columns: dict[str, Any]
    config_hash: str
    logical_run: int
    base_seed: int
    all_seeds: tuple[int, ...]
    simulator_binary: str
    campaign_dir: str
    gui: bool
    max_retries: int
    retry_new_seed: bool
    simulator_output: str


@dataclass(frozen=True)
class TaskArtifact:
    manifest_path: Path
    data_path: Path
    result_filename: str
    extra_columns: dict[str, Any]
    config_hash: str
    seed: int
    retry_attempt: int
    logical_run: int | None
    combination_ordinal: int | None
    task_uuid: str


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _atomic_write_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    tmp_path.write_text(
        json.dumps(_json_safe(document), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp_path, path)


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PogobatchError(f"Configuration not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise PogobatchError(f"Invalid YAML configuration {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise PogobatchError(f"Configuration must be a YAML mapping: {path}")
    return document


def _split_pogobatch_config(
    document: dict[str, Any],
    *,
    warn_legacy: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split scientific Pogosim config from the reserved ``pogobatch`` block.

    ``pogobatch`` is never part of the simulator input.  For backward
    compatibility, the historical top-level ``result_filename_format`` and
    ``result_new_columns`` keys are accepted and folded into the Pogobatch
    settings.  Canonical ``pogobatch`` values take precedence when both forms
    are present.
    """
    scientific = copy.deepcopy(document)
    raw_settings = scientific.pop(POGOBATCH_CONFIG_KEY, {})
    if raw_settings is None:
        raw_settings = {}
    if not isinstance(raw_settings, dict):
        raise PogobatchError(f"{POGOBATCH_CONFIG_KEY} must be a YAML mapping")

    settings = copy.deepcopy(raw_settings)
    legacy_used: list[str] = []
    for key in LEGACY_RESULT_KEYS:
        if key not in scientific:
            continue
        legacy_value = scientific.pop(key)
        legacy_used.append(key)
        if key not in settings:
            settings[key] = copy.deepcopy(legacy_value)
        elif settings[key] != legacy_value:
            logger.warning(
                "Ignoring legacy top-level %s because pogobatch.%s is also set",
                key,
                key,
            )

    if warn_legacy and legacy_used:
        logger.warning(
            "Legacy top-level Pogobatch setting(s) %s are deprecated; move them "
            "under the top-level 'pogobatch:' block",
            ", ".join(legacy_used),
        )

    return scientific, settings


def _pogobatch_task_settings(settings: dict[str, Any]) -> dict[str, Any]:
    """Return only Pogobatch settings that atomic tasks need for result metadata."""
    result: dict[str, Any] = {}
    for key in LEGACY_RESULT_KEYS:
        if key in settings:
            result[key] = copy.deepcopy(settings[key])
    return result


def _pogobatch_rundra_sync_exclude(settings: dict[str, Any]) -> list[str]:
    """Read canonical ``pogobatch.rundra.sync.exclude`` entries."""
    raw_rundra = settings.get("rundra", {})
    if raw_rundra is None:
        return []
    if not isinstance(raw_rundra, dict):
        raise PogobatchError("pogobatch.rundra must be a mapping")

    raw_sync = raw_rundra.get("sync", {})
    if raw_sync is None:
        return []
    if not isinstance(raw_sync, dict):
        raise PogobatchError("pogobatch.rundra.sync must be a mapping")

    raw_exclude = raw_sync.get("exclude", [])
    if raw_exclude is None:
        return []
    if not isinstance(raw_exclude, list) or any(
        not isinstance(item, str) or not item.strip() for item in raw_exclude
    ):
        raise PogobatchError(
            "pogobatch.rundra.sync.exclude must be a list of non-empty strings"
        )
    return [item.strip() for item in raw_exclude]


def _to_dotdict(obj: Any) -> Any:
    if isinstance(obj, dict):
        return DotDict({key: _to_dotdict(value) for key, value in obj.items()})
    if isinstance(obj, list):
        return [_to_dotdict(value) for value in obj]
    return obj


def _get_by_dotted_path(node: Any, dotted: str, sep: str = ".") -> Any:
    current = node
    for raw_part in dotted.split(sep):
        if isinstance(current, list):
            current = current[int(raw_part)]
        else:
            current = current[raw_part]
    return current


def _set_by_path(node: Any, path: tuple[str | int, ...], value: Any) -> None:
    if not path:
        raise PogobatchError("A batch marker cannot replace the configuration root")
    current = node
    for part in path[:-1]:
        current = current[part]
    current[path[-1]] = value


def _get_by_path(node: Any, path: tuple[str | int, ...]) -> Any:
    current = node
    for part in path:
        current = current[part]
    return current


def _path_name(path: tuple[str | int, ...]) -> str:
    return ".".join(str(part) for part in path)


def _normalise_for_filename(node: Any) -> Any:
    if isinstance(node, dict):
        return {key: _normalise_for_filename(value) for key, value in node.items()}
    if isinstance(node, list):
        return [_normalise_for_filename(value) for value in node]
    if isinstance(node, str) and os.path.dirname(node):
        return os.path.splitext(os.path.basename(node))[0]
    return node


def _config_hash(config: dict[str, Any]) -> str:
    content = yaml.safe_dump(config, sort_keys=True, allow_unicode=True)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _contains_batch_marker(node: Any) -> bool:
    if isinstance(node, dict):
        if any(
            marker in node
            for marker in (
                "batch_options",
                "batch_options_range",
                "batch_hierarchical_options",
            )
        ):
            return True
        return any(_contains_batch_marker(value) for value in node.values())
    if isinstance(node, list):
        return any(_contains_batch_marker(value) for value in node)
    return False


def _seed_range(text: str) -> tuple[int, ...]:
    parts = text.split(":")
    if len(parts) != 2:
        raise PogobatchError("Seed range must use inclusive START:STOP syntax")
    try:
        start = int(parts[0])
        stop = int(parts[1])
    except ValueError as exc:
        raise PogobatchError("Seed range endpoints must be integers") from exc
    if start < 0 or stop < 0:
        raise PogobatchError("Seeds must be non-negative")
    if stop < start:
        raise PogobatchError("Seed range stop precedes start")
    return tuple(range(start, stop + 1))


def _seeds_from_rundra_metadata(config: dict[str, Any]) -> tuple[int, ...] | None:
    metadata = config.get(RUNDRA_METADATA_KEY)
    if metadata is None:
        return None
    if not isinstance(metadata, dict) or metadata.get("version") != 1:
        return None
    raw = metadata.get("seeds")
    if raw is None:
        return None
    if type(raw) is int:
        if raw < 0:
            raise PogobatchError("_rundr.seeds must be non-negative")
        return (raw,)
    if isinstance(raw, str):
        return _seed_range(raw)
    raise PogobatchError("_rundr.seeds must be an integer or START:STOP string")


def _random_seed() -> int:
    """Return a non-negative random seed accepted by Pogosim."""
    # Stay in the positive signed 32-bit range, which is portable across the
    # simulator CLI and C/C++ integer parsers while still providing ample entropy.
    return secrets.randbelow(2**31)


def _resolve_seeds(
    config: dict[str, Any],
    runs: int | None,
    seeds: str | None,
    seed: int | None = None,
) -> tuple[int, ...]:
    if seed is not None:
        if seed < 0:
            raise PogobatchError("--seed must be non-negative")
        return (seed,)
    if seeds is not None:
        return _seed_range(seeds)
    if runs is not None:
        if runs <= 0:
            raise PogobatchError("--runs must be greater than zero")
        return tuple(range(runs))
    configured = _seeds_from_rundra_metadata(config)
    if configured is not None:
        return configured
    return (_random_seed(),)


# ---------------------------------------------------------------------------
# Pogosim batch expansion and result semantics
# ---------------------------------------------------------------------------


def expand_batch_options(node: dict[str, Any], dotted_path: str = "") -> list[Any]:
    """Expand either batch_options or batch_options_range."""
    if "batch_options" in node:
        values = node["batch_options"]
        if not isinstance(values, list) or not values:
            raise PogobatchError(
                f"{dotted_path or '<root>'}: batch_options must be a non-empty list"
            )
        return values

    if "batch_options_range" not in node:
        raise KeyError(f"{dotted_path or '<root>'}: no batch options found")

    spec = node["batch_options_range"]
    if not isinstance(spec, dict):
        raise PogobatchError(
            f"{dotted_path or '<root>'}: batch_options_range must be a mapping"
        )

    required = {"start", "stop", "step"}
    missing = sorted(required - set(spec))
    unknown = sorted(
        set(spec) - {"start", "stop", "step", "inclusive", "type"}
    )
    if missing:
        raise PogobatchError(
            f"{dotted_path or '<root>'}: missing batch range fields: {missing}"
        )
    if unknown:
        raise PogobatchError(
            f"{dotted_path or '<root>'}: unknown batch range fields: {unknown}"
        )

    start = spec["start"]
    stop = spec["stop"]
    step = spec["step"]
    inclusive = spec.get("inclusive", False)
    value_type = spec.get("type")

    if any(type(value) not in {int, float} for value in (start, stop, step)):
        raise PogobatchError(
            f"{dotted_path or '<root>'}: start/stop/step must be numeric"
        )
    if step == 0:
        raise PogobatchError(f"{dotted_path or '<root>'}: step must be non-zero")
    if type(inclusive) is not bool:
        raise PogobatchError(
            f"{dotted_path or '<root>'}: inclusive must be true or false"
        )

    if value_type is None:
        int_like = all(float(value).is_integer() for value in (start, stop, step))
        value_type = "int" if int_like else "float"
    if value_type not in {"int", "float"}:
        raise PogobatchError(
            f"{dotted_path or '<root>'}: type must be 'int' or 'float'"
        )

    result: list[Any] = []
    current = float(start)
    stop_f = float(stop)
    step_f = float(step)
    epsilon = 1e-12

    def keep_going(value: float) -> bool:
        if step_f > 0:
            return value <= stop_f + epsilon if inclusive else value < stop_f - epsilon
        return value >= stop_f - epsilon if inclusive else value > stop_f + epsilon

    while keep_going(current):
        result.append(
            int(round(current)) if value_type == "int" else round(current, 12)
        )
        current += step_f

    if not result:
        raise PogobatchError(
            f"{dotted_path or '<root>'}: batch_options_range produced no values"
        )
    return result



def resolve_batch_defaults(config: dict[str, Any]) -> dict[str, Any]:
    """Resolve all batch markers to their configured default values.

    Simple batch nodes (``batch_options`` / ``batch_options_range``) must define
    ``default_option``. Hierarchical batch nodes use the ``default`` mapping from
    ``batch_hierarchical_options``. The returned configuration contains no batch
    markers and is suitable for one atomic ``pogobatch task`` execution.
    """
    resolved, _ = _split_pogobatch_config(config)
    resolved.pop(RUNDRA_METADATA_KEY, None)
    choice_names: dict[str, str] = {}
    alias_values: dict[str, str] = {}

    def recurse(node: Any, path: tuple[str | int, ...] = ()) -> Any:
        if isinstance(node, dict):
            present = [
                marker
                for marker in (
                    "batch_options",
                    "batch_options_range",
                    "batch_hierarchical_options",
                )
                if marker in node
            ]
            if len(present) > 1:
                raise PogobatchError(
                    f"{_path_name(path) or '<root>'}: only one batch marker is allowed"
                )

            if "batch_options" in node or "batch_options_range" in node:
                if "default_option" not in node:
                    raise PogobatchError(
                        f"{_path_name(path) or '<root>'}: batch parameter has no "
                        "default_option; specify a resolved configuration instead"
                    )
                return copy.deepcopy(node["default_option"])

            if "batch_hierarchical_options" in node:
                mapping = node["batch_hierarchical_options"]
                if not isinstance(mapping, dict):
                    raise PogobatchError(
                        f"{_path_name(path) or '<root>'}: "
                        "batch_hierarchical_options must be a mapping"
                    )
                default = mapping.get("default")
                if not isinstance(default, dict):
                    raise PogobatchError(
                        f"{_path_name(path) or '<root>'}: hierarchical batch parameter "
                        "has no 'default' mapping"
                    )

                result = {
                    key: recurse(value, (*path, key))
                    for key, value in node.items()
                    if key != "batch_hierarchical_options"
                }
                for key, value in copy.deepcopy(default).items():
                    result[key] = recurse(value, (*path, key))

                dotted = _path_name(path)
                choice_names[dotted] = "default"

                alias = mapping.get("name")
                if alias is not None:
                    if not isinstance(alias, str) or not alias.strip():
                        raise PogobatchError(
                            f"{dotted or '<root>'}: hierarchical name must be a "
                            "non-empty string"
                        )
                    # Match expand_combinations(): expose the selected hierarchical
                    # choice through the alias for result filename/column semantics.
                    # Defer the top-level write until recursion completes.
                    alias_values[alias] = "default"
                    choice_names[alias] = "default"

                return result

            return {
                key: recurse(value, (*path, key))
                for key, value in node.items()
            }

        if isinstance(node, list):
            return [recurse(value, (*path, index)) for index, value in enumerate(node)]

        return copy.deepcopy(node)

    resolved = recurse(resolved)
    resolved.update(alias_values)
    if choice_names:
        resolved.setdefault("_batch_choice_names", {}).update(choice_names)

    if _contains_batch_marker(resolved):
        raise PogobatchError("Internal error: batch defaults were not fully resolved")
    return resolved


def expand_combinations(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Materialize all Pogobatch sweep combinations deterministically."""
    scientific, _ = _split_pogobatch_config(config)
    scientific.pop(RUNDRA_METADATA_KEY, None)

    simple_paths: list[tuple[str | int, ...]] = []
    simple_values: list[list[Any]] = []
    hierarchical_paths: list[tuple[str | int, ...]] = []
    hierarchical_maps: list[dict[str, Any]] = []
    hierarchical_names: list[list[str]] = []
    hierarchical_aliases: list[str | None] = []

    def recurse(node: Any, path: tuple[str | int, ...] = ()) -> None:
        if isinstance(node, dict):
            present = [
                marker
                for marker in (
                    "batch_options",
                    "batch_options_range",
                    "batch_hierarchical_options",
                )
                if marker in node
            ]
            if len(present) > 1:
                raise PogobatchError(
                    f"{_path_name(path) or '<root>'}: only one batch marker is allowed"
                )

            if "batch_options" in node or "batch_options_range" in node:
                simple_paths.append(path)
                simple_values.append(
                    expand_batch_options(node, dotted_path=_path_name(path))
                )
                return

            if "batch_hierarchical_options" in node:
                mapping = node["batch_hierarchical_options"]
                if not isinstance(mapping, dict):
                    raise PogobatchError(
                        f"{_path_name(path)}: batch_hierarchical_options must be a mapping"
                    )
                names = [
                    key for key in mapping if key not in {"default", "name"}
                ]
                if not names:
                    raise PogobatchError(
                        f"{_path_name(path)}: hierarchical batch has no alternatives"
                    )
                if any(not isinstance(mapping[name], dict) for name in names):
                    raise PogobatchError(
                        f"{_path_name(path)}: hierarchical alternatives must be mappings"
                    )
                alias = mapping.get("name")
                if alias is not None and (
                    not isinstance(alias, str) or not alias.strip()
                ):
                    raise PogobatchError(
                        f"{_path_name(path)}: hierarchical name must be a non-empty string"
                    )
                hierarchical_paths.append(path)
                hierarchical_maps.append(mapping)
                hierarchical_names.append(names)
                hierarchical_aliases.append(alias)
                for key, value in node.items():
                    if key != "batch_hierarchical_options":
                        recurse(value, (*path, key))
                return

            for key, value in node.items():
                recurse(value, (*path, key))
        elif isinstance(node, list):
            for index, value in enumerate(node):
                recurse(value, (*path, index))

    recurse(scientific)

    factors: list[list[Any]] = [*simple_values, *hierarchical_names]
    if not factors:
        return [scientific]

    combinations: list[dict[str, Any]] = []
    for product in itertools.product(*factors):
        effective = copy.deepcopy(scientific)
        choice_names: dict[str, str] = {}
        offset = 0

        for path, value in zip(
            simple_paths,
            product[: len(simple_paths)],
            strict=True,
        ):
            _set_by_path(effective, path, copy.deepcopy(value))
        offset += len(simple_paths)

        for path, mapping, choice_name, alias in zip(
            hierarchical_paths,
            hierarchical_maps,
            product[offset:],
            hierarchical_aliases,
            strict=True,
        ):
            parent = _get_by_path(effective, path)
            if not isinstance(parent, dict):
                raise AssertionError("validated hierarchical batch changed shape")
            parent.pop("batch_hierarchical_options", None)
            parent.update(copy.deepcopy(mapping[choice_name]))
            dotted = _path_name(path)
            choice_names[dotted] = choice_name
            if alias:
                effective[alias] = choice_name
                choice_names[alias] = choice_name

        if choice_names:
            effective.setdefault("_batch_choice_names", {}).update(choice_names)
        combinations.append(effective)

    return combinations



def count_combinations(config: dict[str, Any]) -> int:
    """Return the number of Pogobatch parameter combinations without building them.

    This deliberately counts independent sweep factors only. It avoids the old
    cluster path's Cartesian-product materialization, which duplicated the full
    scientific YAML once per combination before Rundra even saw the campaign.
    """
    scientific, _ = _split_pogobatch_config(config)
    scientific.pop(RUNDRA_METADATA_KEY, None)
    factor_sizes: list[int] = []

    def recurse(node: Any, path: tuple[str | int, ...] = ()) -> None:
        if isinstance(node, dict):
            present = [
                key
                for key in (
                    "batch_options",
                    "batch_options_range",
                    "batch_hierarchical_options",
                )
                if key in node
            ]
            if len(present) > 1:
                raise PogobatchError(
                    f"{_path_name(path) or '<root>'}: only one batch marker is allowed"
                )

            if "batch_options" in node or "batch_options_range" in node:
                factor_sizes.append(
                    len(expand_batch_options(node, dotted_path=_path_name(path)))
                )
                return

            if "batch_hierarchical_options" in node:
                mapping = node["batch_hierarchical_options"]
                if not isinstance(mapping, dict):
                    raise PogobatchError(
                        f"{_path_name(path)}: batch_hierarchical_options must be a mapping"
                    )
                names = [name for name in mapping if name not in {"default", "name"}]
                if not names:
                    raise PogobatchError(
                        f"{_path_name(path)}: hierarchical batch has no alternatives"
                    )
                if any(not isinstance(mapping[name], dict) for name in names):
                    raise PogobatchError(
                        f"{_path_name(path)}: hierarchical alternatives must be mappings"
                    )
                factor_sizes.append(len(names))
                # Rundra, like the local Pogobatch expander, still considers
                # independent sweep markers in sibling fields of this node.
                for key, value in node.items():
                    if key != "batch_hierarchical_options":
                        recurse(value, (*path, key))
                return

            for key, value in node.items():
                recurse(value, (*path, key))
        elif isinstance(node, list):
            for index, value in enumerate(node):
                recurse(value, (*path, index))

    recurse(scientific)
    count = 1
    for size in factor_sizes:
        count *= size
    return count


def make_rundra_sweep_config(config: dict[str, Any]) -> dict[str, Any]:
    """Make a compact Rundra-ready copy of a Pogobatch scientific config.

    Rundra already natively expands ``batch_options``, ``batch_options_range``
    and ``batch_hierarchical_options`` when ``_rundr.version == 1``. Therefore
    the generated cluster config should remain O(size of the original YAML), not
    O(number of Cartesian-product combinations).

    Hierarchical choices need one small Pogobatch-only marker because Rundra's
    effective scientific config contains the selected mapping but not the old
    Pogobatch top-level ``name`` alias used by ``result_new_columns`` and some
    filename formats. ``pogobatch task`` removes these markers before invoking
    the simulator and uses them only to reconstruct result metadata.
    """
    scientific, pogobatch_settings = _split_pogobatch_config(config)
    result = copy.deepcopy(scientific)
    task_settings = _pogobatch_task_settings(pogobatch_settings)
    if task_settings:
        result[POGOBATCH_CONFIG_KEY] = task_settings

    metadata = result.get(RUNDRA_METADATA_KEY)
    if metadata is None:
        result[RUNDRA_METADATA_KEY] = {"version": 1}
    elif not isinstance(metadata, dict):
        raise PogobatchError("_rundr must be a mapping when present")
    elif metadata.get("version") != 1:
        raise PogobatchError("Generated Rundra sweep requires _rundr.version: 1")

    def recurse(node: Any, path: tuple[str | int, ...] = ()) -> None:
        if isinstance(node, dict):
            mapping = node.get("batch_hierarchical_options")
            if mapping is not None:
                if not isinstance(mapping, dict):
                    raise PogobatchError(
                        f"{_path_name(path)}: batch_hierarchical_options must be a mapping"
                    )
                alias = mapping.get("name")
                if alias is not None and (
                    not isinstance(alias, str) or not alias.strip()
                ):
                    raise PogobatchError(
                        f"{_path_name(path)}: hierarchical name must be a non-empty string"
                    )
                dotted = _path_name(path)
                for choice_name, choice in mapping.items():
                    if choice_name in {"default", "name"}:
                        continue
                    if not isinstance(choice, dict):
                        raise PogobatchError(
                            f"{dotted}: hierarchical alternative {choice_name!r} must be a mapping"
                        )
                    if POGOBATCH_CHOICE_KEY in choice:
                        raise PogobatchError(
                            f"{dotted}: reserved key {POGOBATCH_CHOICE_KEY!r} is already present"
                        )
                    choice[POGOBATCH_CHOICE_KEY] = {
                        "path": dotted,
                        "choice": choice_name,
                        **({"alias": alias} if alias else {}),
                    }
                for key, value in node.items():
                    if key != "batch_hierarchical_options":
                        recurse(value, (*path, key))
                return

            for key, value in node.items():
                recurse(value, (*path, key))
        elif isinstance(node, list):
            for index, value in enumerate(node):
                recurse(value, (*path, index))

    recurse(result)
    return result


def _extract_rundra_choice_metadata(
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, str], dict[str, str]]:
    """Strip generated hierarchical markers and recover alias/choice metadata."""
    scientific = copy.deepcopy(config)
    aliases: dict[str, str] = {}
    choice_names: dict[str, str] = {}

    def recurse(node: Any, path: tuple[str | int, ...] = ()) -> None:
        if isinstance(node, dict):
            marker = node.pop(POGOBATCH_CHOICE_KEY, None)
            if marker is not None:
                if not isinstance(marker, dict):
                    raise PogobatchError(
                        f"{_path_name(path)}: invalid internal Pogobatch choice marker"
                    )
                choice = marker.get("choice")
                declared_path = marker.get("path")
                alias = marker.get("alias")
                if not isinstance(choice, str) or not choice:
                    raise PogobatchError(
                        f"{_path_name(path)}: invalid internal hierarchical choice"
                    )
                if not isinstance(declared_path, str) or not declared_path:
                    declared_path = _path_name(path)
                choice_names[declared_path] = choice
                if alias is not None:
                    if not isinstance(alias, str) or not alias:
                        raise PogobatchError(
                            f"{_path_name(path)}: invalid internal hierarchical alias"
                        )
                    aliases[alias] = choice
                    choice_names[alias] = choice

            for key, value in list(node.items()):
                recurse(value, (*path, key))
        elif isinstance(node, list):
            for index, value in enumerate(node):
                recurse(value, (*path, index))

    recurse(scientific)
    return scientific, aliases, choice_names

def compute_result_filename(
    config: dict[str, Any],
    pogobatch_settings: dict[str, Any] | None = None,
) -> str:
    settings = pogobatch_settings or {}
    fmt = settings.get("result_filename_format", config.get("result_filename_format"))
    if not fmt:
        return "result.feather"
    if not isinstance(fmt, str):
        raise PogobatchError("result_filename_format must be a string")

    dot_config = _to_dotdict(_normalise_for_filename(copy.deepcopy(config)))
    try:
        return fmt.format_map(dot_config)
    except Exception as exc:
        raise PogobatchError(
            f"Could not format result_filename_format {fmt!r}: {exc}"
        ) from exc


def compute_extra_columns(
    config: dict[str, Any],
    pogobatch_settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    settings = pogobatch_settings or {}
    requested = settings.get("result_new_columns", config.get("result_new_columns", []))
    if requested is None:
        return result
    if not isinstance(requested, list) or any(
        not isinstance(path, str) for path in requested
    ):
        raise PogobatchError("result_new_columns must be a list of dotted paths")

    choice_names = config.get("_batch_choice_names", {})
    if not isinstance(choice_names, dict):
        choice_names = {}

    for path in requested:
        try:
            raw = _get_by_dotted_path(config, path)
        except (KeyError, IndexError, ValueError, TypeError) as exc:
            raise PogobatchError(
                f"result_new_columns path not found: {path}"
            ) from exc

        if isinstance(raw, dict):
            raw = choice_names.get(path)
        elif isinstance(raw, str) and os.path.dirname(raw):
            raw = os.path.splitext(os.path.basename(raw))[0]
        result[path] = raw
    return result


def build_combinations(config: dict[str, Any]) -> list[Combination]:
    scientific, pogobatch_settings = _split_pogobatch_config(config)
    result: list[Combination] = []
    for ordinal, effective in enumerate(expand_combinations(scientific)):
        result.append(
            Combination(
                ordinal=ordinal,
                config=effective,
                result_filename=compute_result_filename(
                    effective, pogobatch_settings
                ),
                extra_columns=compute_extra_columns(
                    effective, pogobatch_settings
                ),
                config_hash=_config_hash(effective),
            )
        )
    return result


def _filter_combinations(
    combinations: Sequence[Combination], only_output: str | None
) -> list[Combination]:
    if not only_output:
        return list(combinations)
    requested_base = os.path.basename(only_output)
    filtered = [
        combination
        for combination in combinations
        if combination.result_filename == only_output
        or os.path.basename(combination.result_filename) == requested_base
    ]
    if not filtered:
        raise PogobatchError(f"No combination matched --only-output={only_output}")
    return filtered


# ---------------------------------------------------------------------------
# Atomic task execution
# ---------------------------------------------------------------------------


def _extract_task_payload(
    document: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = document.get(RUNDRA_WRAPPER_KEY)
    if payload is None:
        return document, {}
    if not isinstance(payload, dict):
        raise PogobatchError(f"{RUNDRA_WRAPPER_KEY} must be a mapping")
    if "batch_hierarchical_options" in payload or "batch_options" in payload:
        raise PogobatchError(
            "Received an unresolved Pogobatch/Rundra wrapper; Rundra must expand it "
            "before invoking 'pogobatch task'"
        )
    config = payload.get("config")
    metadata = payload.get("metadata", {})
    if not isinstance(config, dict):
        raise PogobatchError(
            f"Resolved {RUNDRA_WRAPPER_KEY} must contain a configuration mapping"
        )
    if not isinstance(metadata, dict):
        raise PogobatchError(
            f"Resolved {RUNDRA_WRAPPER_KEY}.metadata must be a mapping"
        )
    return copy.deepcopy(config), copy.deepcopy(metadata)


def _prepare_runtime_config(
    config: dict[str, Any], output_dir: Path
) -> tuple[dict[str, Any], Path]:
    runtime = copy.deepcopy(config)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    data_basename = "data.feather"
    if "data_filename" in runtime:
        data_basename = os.path.basename(str(runtime["data_filename"]))
        runtime["data_filename"] = str(frames_dir / data_basename)
    if "console_filename" in runtime:
        runtime["console_filename"] = str(
            frames_dir / os.path.basename(str(runtime["console_filename"]))
        )
    if "frames_name" in runtime:
        runtime["frames_name"] = str(
            frames_dir / os.path.basename(str(runtime["frames_name"]))
        )

    return runtime, frames_dir / data_basename


_SIMULATOR_LOG_LEVEL_RE = re.compile(
    r"\[(trace|debug|info|warning|warn|error|critical|fatal)\]",
    re.IGNORECASE,
)


def _emit_simulator_output(output: str, mode: str) -> None:
    """Emit simulator output according to Pogobatch's verbosity policy."""
    if not output:
        return

    warning_levels = {"warning", "warn", "error", "critical", "fatal"}
    error_levels = {"error", "critical", "fatal"}

    for line in output.splitlines():
        match = _SIMULATOR_LOG_LEVEL_RE.search(line)
        level = match.group(1).lower() if match else None

        if mode == "verbose":
            print(line, file=sys.stderr)
        elif mode == "normal" and level in warning_levels:
            print(line, file=sys.stderr)
        elif mode == "quiet" and level in error_levels:
            print(line, file=sys.stderr)


def _launch_simulator(
    config_path: Path,
    simulator_binary: str,
    seed: int,
    gui: bool,
    simulator_output: str = "normal",
) -> None:
    command = [
        simulator_binary,
        "-c",
        str(config_path),
        "-nr",
        "-q",
        "--seed",
        str(seed),
    ]
    if not gui:
        command.append("-g")
    logger.debug("Executing: %s", shlex.join(command))

    completed = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    _emit_simulator_output(completed.stdout or "", simulator_output)
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(
            completed.returncode,
            command,
            output=completed.stdout,
        )


def execute_task(
    config_path: Path,
    simulator_binary: str,
    seed: int,
    output_dir: Path,
    *,
    gui: bool = False,
    logical_run: int | None = None,
    retry_attempt: int = 0,
    metadata_override: dict[str, Any] | None = None,
    simulator_output: str = "normal",
) -> Path:
    """Execute one configuration/seed, resolving batch defaults when needed."""
    if seed < 0:
        raise PogobatchError("Seed must be non-negative")

    source_document = _load_yaml(config_path)
    resolved_config, wrapper_metadata = _extract_task_payload(source_document)
    resolved_config, pogobatch_settings = _split_pogobatch_config(
        resolved_config, warn_legacy=True
    )
    # A direct task may receive a normal batch configuration. Resolve every
    # batch parameter through its declared default before invoking the simulator.
    if _contains_batch_marker(resolved_config):
        resolved_config = resolve_batch_defaults(resolved_config)
    else:
        # _rundr is orchestration metadata, never part of the simulator input.
        resolved_config.pop(RUNDRA_METADATA_KEY, None)

    # Compact Rundra sweeps carry tiny internal markers only for hierarchical
    # choice labels. Strip them before the simulator sees the configuration and
    # reconstruct the old Pogobatch alias view solely for result metadata.
    resolved_config, rundra_aliases, rundra_choice_names = (
        _extract_rundra_choice_metadata(resolved_config)
    )
    metadata_config = copy.deepcopy(resolved_config)
    if rundra_aliases:
        metadata_config.update(rundra_aliases)
    existing_choice_names = metadata_config.get("_batch_choice_names", {})
    if not isinstance(existing_choice_names, dict):
        existing_choice_names = {}
    if rundra_choice_names:
        metadata_config["_batch_choice_names"] = {
            **existing_choice_names,
            **rundra_choice_names,
        }

    metadata = copy.deepcopy(wrapper_metadata)
    if metadata_override:
        metadata.update(copy.deepcopy(metadata_override))

    # Pogobatch/Rundra bookkeeping is metadata only and must never reach the
    # simulator's YAML parser.
    resolved_config.pop("_batch_choice_names", None)
    resolved_config.pop(POGOBATCH_CONFIG_KEY, None)
    resolved_config.pop(RUNDRA_METADATA_KEY, None)

    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_config, data_path = _prepare_runtime_config(resolved_config, output_dir)
    runtime_config_path = output_dir / "pogobatch_effective.yaml"
    runtime_config_path.write_text(
        yaml.safe_dump(runtime_config, sort_keys=False), encoding="utf-8"
    )

    _launch_simulator(
        runtime_config_path,
        simulator_binary,
        seed,
        gui,
        simulator_output=simulator_output,
    )

    if not data_path.exists():
        raise PogobatchError(
            f"Simulation completed but produced no Feather data file: {data_path}"
        )

    result_filename = metadata.get("result_filename")
    if not isinstance(result_filename, str) or not result_filename:
        result_filename = compute_result_filename(
            metadata_config, pogobatch_settings
        )

    extra_columns = metadata.get("extra_columns")
    if not isinstance(extra_columns, dict):
        extra_columns = compute_extra_columns(
            metadata_config, pogobatch_settings
        )

    config_hash = metadata.get("config_hash")
    if not isinstance(config_hash, str) or not config_hash:
        config_hash = _config_hash(resolved_config)

    combination_ordinal = metadata.get("combination_ordinal")
    if type(combination_ordinal) is not int:
        combination_ordinal = None

    task_uuid = uuid.uuid4().hex
    manifest = {
        "schema_version": TASK_MANIFEST_VERSION,
        "task_uuid": task_uuid,
        "seed": seed,
        "logical_run": logical_run,
        "retry_attempt": retry_attempt,
        "combination_ordinal": combination_ordinal,
        "config_hash": config_hash,
        "result_filename": result_filename,
        "extra_columns": extra_columns,
        "batch_choices": rundra_choice_names,
        "data_file": str(data_path.relative_to(output_dir)),
        # Keep task sidecars compact. The full original campaign configuration
        # is embedded once in the merged Feather metadata, not once per Task.
        "pogosim_version": __version__,
    }
    manifest_path = output_dir / TASK_MANIFEST_NAME
    _atomic_write_json(manifest_path, manifest)
    return manifest_path


# ---------------------------------------------------------------------------
# Local batch execution
# ---------------------------------------------------------------------------


def _replacement_seed(
    base_seed: int,
    logical_run: int,
    all_seeds: tuple[int, ...],
    attempt: int,
) -> int:
    if attempt == 0:
        return base_seed
    count = len(all_seeds)
    if all_seeds == tuple(range(count)):
        return logical_run + attempt * count
    return max(all_seeds) + 1 + (attempt - 1) * count + logical_run


def _run_local_task_worker(spec: LocalTaskSpec) -> dict[str, Any]:
    campaign_dir = Path(spec.campaign_dir)
    last_error: str | None = None

    for attempt in range(spec.max_retries + 1):
        seed = (
            _replacement_seed(
                spec.base_seed,
                spec.logical_run,
                spec.all_seeds,
                attempt,
            )
            if spec.retry_new_seed
            else spec.base_seed
        )
        attempt_dir = (
            campaign_dir
            / f"combo_{spec.combination_ordinal:06d}"
            / f"run_{spec.logical_run:06d}_seed_{seed}_try_{attempt}"
        )
        shutil.rmtree(attempt_dir, ignore_errors=True)
        attempt_dir.mkdir(parents=True, exist_ok=True)
        config_path = attempt_dir / "input_config.yaml"
        config_path.write_text(
            yaml.safe_dump(spec.config, sort_keys=False), encoding="utf-8"
        )

        try:
            manifest_path = execute_task(
                config_path,
                spec.simulator_binary,
                seed,
                attempt_dir,
                gui=spec.gui,
                logical_run=spec.logical_run,
                retry_attempt=attempt,
                metadata_override={
                    "combination_ordinal": spec.combination_ordinal,
                    "config_hash": spec.config_hash,
                    "result_filename": spec.result_filename,
                    "extra_columns": spec.extra_columns,
                },
                simulator_output=spec.simulator_output,
            )
            return {
                "ok": True,
                "manifest": str(manifest_path),
                "combination_ordinal": spec.combination_ordinal,
                "logical_run": spec.logical_run,
                "seed": seed,
                "retry_attempt": attempt,
            }
        except subprocess.CalledProcessError as exc:
            last_error = f"simulator exited with status {exc.returncode}"
            logger.warning(
                "Combination %d run %d seed %d crashed on attempt %d/%d",
                spec.combination_ordinal,
                spec.logical_run,
                seed,
                attempt + 1,
                spec.max_retries + 1,
            )
            shutil.rmtree(attempt_dir, ignore_errors=True)
            if attempt == spec.max_retries:
                break
        except Exception as exc:  # noqa: BLE001 - worker must return structured failure
            last_error = str(exc)
            shutil.rmtree(attempt_dir, ignore_errors=True)
            break

    return {
        "ok": False,
        "combination_ordinal": spec.combination_ordinal,
        "logical_run": spec.logical_run,
        "seed": spec.base_seed,
        "error": last_error or "unknown task failure",
    }


def _progress_iterator(iterable: Any, total: int, enabled: bool) -> Any:
    if not enabled:
        return iterable
    try:
        from tqdm.auto import tqdm
    except ImportError as exc:
        raise PogobatchError(
            "--progress requires tqdm; install it with 'python -m pip install tqdm'"
        ) from exc
    return tqdm(iterable, total=total, unit="task", desc="Pogosim", dynamic_ncols=True)


def _execute_local_specs(
    specs: list[LocalTaskSpec],
    backend: str,
    jobs: int,
    *,
    progress: bool = False,
) -> list[dict[str, Any]]:
    if backend == "sequential":
        return [
            _run_local_task_worker(spec)
            for spec in _progress_iterator(specs, len(specs), progress)
        ]

    if backend == "multiprocessing":
        with Pool(processes=jobs) as pool:
            results = pool.imap_unordered(_run_local_task_worker, specs)
            return list(_progress_iterator(results, len(specs), progress))

    if backend == "ray":
        try:
            import ray
        except ImportError as exc:
            raise PogobatchError("Ray is not installed") from exc
        ray.init(ignore_reinit_error=True)
        try:
            remote_worker = ray.remote(_run_local_task_worker)
            pending = [remote_worker.remote(spec) for spec in specs]
            results: list[dict[str, Any]] = []
            iterator = range(len(pending))
            for _ in _progress_iterator(iterator, len(pending), progress):
                ready, pending = ray.wait(pending, num_returns=1)
                results.append(ray.get(ready[0]))
            return results
        finally:
            ray.shutdown()

    raise PogobatchError(f"Unknown local backend: {backend}")


# ---------------------------------------------------------------------------
# Merge task shards
# ---------------------------------------------------------------------------


def _load_task_artifact(manifest_path: Path) -> TaskArtifact:
    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PogobatchError(f"Invalid task manifest {manifest_path}: {exc}") from exc

    if document.get("schema_version") not in SUPPORTED_TASK_MANIFEST_VERSIONS:
        raise PogobatchError(
            f"Unsupported task manifest version in {manifest_path}: "
            f"{document.get('schema_version')!r}"
        )

    data_file = document.get("data_file")
    if not isinstance(data_file, str) or not data_file:
        raise PogobatchError(f"Task manifest has no data_file: {manifest_path}")
    data_path = manifest_path.parent / data_file
    if not data_path.exists():
        raise PogobatchError(
            f"Task manifest references missing Feather file: {data_path}"
        )

    result_filename = document.get("result_filename")
    if not isinstance(result_filename, str) or not result_filename:
        raise PogobatchError(
            f"Task manifest has no result_filename: {manifest_path}"
        )

    extra_columns = document.get("extra_columns", {})
    if not isinstance(extra_columns, dict):
        raise PogobatchError(
            f"Task manifest extra_columns is not a mapping: {manifest_path}"
        )

    config_hash = document.get("config_hash")
    if not isinstance(config_hash, str) or not config_hash:
        raise PogobatchError(f"Task manifest has no config_hash: {manifest_path}")

    seed = document.get("seed")
    retry_attempt = document.get("retry_attempt", 0)
    if type(seed) is not int or type(retry_attempt) is not int:
        raise PogobatchError(f"Invalid seed/retry metadata: {manifest_path}")

    logical_run = document.get("logical_run")
    if type(logical_run) is not int:
        logical_run = None
    combination_ordinal = document.get("combination_ordinal")
    if type(combination_ordinal) is not int:
        combination_ordinal = None

    task_uuid = document.get("task_uuid")
    if not isinstance(task_uuid, str) or not task_uuid:
        task_uuid = hashlib.sha256(str(manifest_path).encode("utf-8")).hexdigest()

    return TaskArtifact(
        manifest_path=manifest_path,
        data_path=data_path,
        result_filename=result_filename,
        extra_columns=extra_columns,
        config_hash=config_hash,
        seed=seed,
        retry_attempt=retry_attempt,
        logical_run=logical_run,
        combination_ordinal=combination_ordinal,
        task_uuid=task_uuid,
    )


def discover_task_artifacts(roots: Sequence[Path]) -> list[TaskArtifact]:
    manifests: list[Path] = []
    for root in roots:
        if root.is_file() and root.name == TASK_MANIFEST_NAME:
            manifests.append(root)
        elif root.is_dir():
            manifests.extend(root.rglob(TASK_MANIFEST_NAME))
        else:
            raise PogobatchError(f"Merge input does not exist: {root}")

    if not manifests:
        raise PogobatchError(
            "No pogobatch_task.json manifests found. If these are Rundra results, "
            "fetch them with ordinary files (for example: rundr fetch RUN_ID --mode copy) "
            "and ensure the Rundra experiment declares both the task manifest and Feather output."
        )

    artifacts = [_load_task_artifact(path) for path in sorted(set(manifests))]
    seen: set[str] = set()
    for artifact in artifacts:
        if artifact.task_uuid in seen:
            raise PogobatchError(
                f"Duplicate task UUID discovered while merging: {artifact.task_uuid}"
            )
        seen.add(artifact.task_uuid)
    return artifacts


def _derived_run_indices(artifacts: Sequence[TaskArtifact]) -> dict[str, int]:
    result: dict[str, int] = {}
    by_config: dict[str, list[TaskArtifact]] = {}
    for artifact in artifacts:
        by_config.setdefault(artifact.config_hash, []).append(artifact)

    for items in by_config.values():
        items.sort(
            key=lambda item: (
                item.seed,
                item.retry_attempt,
                str(item.manifest_path),
            )
        )
        for index, artifact in enumerate(items):
            result[artifact.task_uuid] = (
                artifact.logical_run
                if artifact.logical_run is not None
                else index
            )
    return result


def _constant_array(value: Any, length: int) -> pa.Array:
    if length == 0:
        if value is None:
            return pa.array([], type=pa.null())
        return pa.array([], type=pa.scalar(value).type)
    return pa.array([value] * length)


def _set_table_column(table: pa.Table, name: str, value: Any) -> pa.Table:
    array = _constant_array(value, table.num_rows)
    index = table.schema.get_field_index(name)
    if index >= 0:
        return table.set_column(index, name, array)
    return table.append_column(name, array)


def _concat_tables(tables: list[pa.Table]) -> pa.Table:
    if not tables:
        raise PogobatchError("Internal error: no tables to concatenate")
    if len(tables) == 1:
        return tables[0]
    try:
        return pa.concat_tables(tables, promote_options="default")
    except TypeError:
        return pa.concat_tables(tables, promote=True)


def _resolve_merge_output(output_dir: Path, logical_name: str) -> Path:
    logical = Path(logical_name)
    if logical.is_absolute():
        logger.warning(
            "Absolute result filename %s is rebased under merge output directory",
            logical_name,
        )
        logical = Path(logical.name)
    return output_dir / logical


def merge_task_artifacts(
    artifacts: Sequence[TaskArtifact],
    output_dir: Path,
    *,
    configuration_path: Path | None = None,
    append: bool = False,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_indices = _derived_run_indices(artifacts)
    groups: dict[str, list[TaskArtifact]] = {}
    for artifact in artifacts:
        groups.setdefault(artifact.result_filename, []).append(artifact)

    configuration_text: str | None = None
    if configuration_path is not None:
        configuration_text = configuration_path.read_text(encoding="utf-8")

    outputs: list[Path] = []
    for logical_name in sorted(groups):
        items = groups[logical_name]
        items.sort(
            key=lambda item: (
                item.combination_ordinal
                if item.combination_ordinal is not None
                else 10**12,
                run_indices[item.task_uuid],
                item.seed,
                str(item.manifest_path),
            )
        )

        tables: list[pa.Table] = []
        for artifact in items:
            table = feather.read_table(artifact.data_path, memory_map=True)
            table = _set_table_column(
                table, "run", run_indices[artifact.task_uuid]
            )
            table = _set_table_column(table, "seed", artifact.seed)
            table = _set_table_column(
                table, "retry_attempt", artifact.retry_attempt
            )
            for column, value in artifact.extra_columns.items():
                table = _set_table_column(table, column, value)
            tables.append(table)

        combined = _concat_tables(tables)
        final_path = _resolve_merge_output(output_dir, logical_name)
        final_path.parent.mkdir(parents=True, exist_ok=True)

        if append and final_path.exists():
            old_table = feather.read_table(final_path, memory_map=True)
            combined = _concat_tables([old_table, combined])

        metadata = dict(combined.schema.metadata or {})
        if configuration_text is not None:
            metadata[b"configuration"] = configuration_text.encode("utf-8")
        metadata[b"pogobatch_manifest"] = json.dumps(
            {
                "schema_version": 1,
                "task_count": len(items),
                "config_hashes": sorted({item.config_hash for item in items}),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        combined = combined.replace_schema_metadata(metadata)
        feather.write_feather(combined, final_path)
        outputs.append(final_path)
        logger.info("Merged %d task shard(s) -> %s", len(items), final_path)

    return outputs


# ---------------------------------------------------------------------------
# Rundra integration
# ---------------------------------------------------------------------------


def make_rundra_wrapper_config(
    combinations: Sequence[Combination],
) -> dict[str, Any]:
    """Create a one-factor Rundra sweep over already-resolved Pogobatch combos.

    Pogobatch remains authoritative for its batch semantics.  Rundra receives a
    deterministic hierarchical factor whose alternatives carry one fully resolved
    scientific config plus the metadata needed by the generic merger.
    """
    alternatives: dict[str, Any] = {
        "name": "pogobatch_combination",
    }
    for combination in combinations:
        alternatives[f"combo_{combination.ordinal:06d}"] = {
            "config": combination.config,
            "metadata": {
                "combination_ordinal": combination.ordinal,
                "config_hash": combination.config_hash,
                "result_filename": combination.result_filename,
                "extra_columns": combination.extra_columns,
            },
        }

    return {
        RUNDRA_METADATA_KEY: {"version": 1},
        RUNDRA_WRAPPER_KEY: {
            "batch_hierarchical_options": alternatives,
        },
    }


def make_rundra_experiment(
    *,
    name: str,
    simulator_binary: str,
    task_launcher: Sequence[str],
    cpus_per_task: int,
    gpus_per_task: int,
    memory: str,
    walltime: str,
    container_image: str | None,
    container_gpu: bool,
    sync_exclude: Sequence[str] = (),
) -> dict[str, Any]:
    command = [
        *task_launcher,
        "task",
        "-c",
        "{config}",
        "-S",
        simulator_binary,
        "--seed",
        "{seed}",
        "-o",
        "/workspace/output",
    ]
    experiment: dict[str, Any] = {
        "version": 1,
        "experiment": {"name": name},
        "command": {"argv": command},
        "resources": {
            "nodes": 1,
            "tasks": 1,
            "cpus_per_task": cpus_per_task,
            "gpus_per_task": gpus_per_task,
            "memory": memory,
            "walltime": walltime,
        },
        "outputs": {
            "include": [
                TASK_MANIFEST_NAME,
                "frames/**",
            ]
        },
    }
    if sync_exclude:
        experiment["sync"] = {"exclude": list(sync_exclude)}
    if container_image:
        experiment["container"] = {
            "image": container_image,
            "gpu": container_gpu,
        }
    return experiment


def _apptainer_definition_context(
    definition_path: Path,
    source_root: Path,
) -> list[str]:
    """Return source-root-relative paths referenced by an Apptainer ``%files`` block.

    Rundra project v4+ requires an explicit definition build context. Pogosim's
    shipped definition files use ordinary ``%files`` entries whose first token is
    a source path relative to the repository root. Keep the parser intentionally
    narrow and fail rather than silently omit an unfamiliar path.
    """
    source_root = source_root.expanduser().resolve()
    definition_path = definition_path.expanduser().resolve()
    try:
        definition_relative = definition_path.relative_to(source_root)
    except ValueError as exc:
        raise PogobatchError(
            f"Apptainer definition must be inside the Rundra source root {source_root}: "
            f"{definition_path}"
        ) from exc

    in_files = False
    includes: list[str] = []
    seen: set[str] = set()
    for raw_line in definition_path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if stripped.startswith("%"):
            in_files = stripped.split(maxsplit=1)[0] == "%files"
            continue
        if not in_files or not stripped or stripped.startswith("#"):
            continue
        try:
            fields = shlex.split(stripped, comments=True, posix=True)
        except ValueError as exc:
            raise PogobatchError(
                f"Could not parse %files entry in {definition_path}: {raw_line!r}"
            ) from exc
        if not fields:
            continue
        raw_source = fields[0]
        candidate = Path(raw_source)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise PogobatchError(
                f"Unsupported non-relative %files source {raw_source!r} in "
                f"{definition_path}; automatic Rundra definition preparation "
                "requires repository-relative sources"
            )
        while candidate.parts and candidate.parts[0] == ".":
            candidate = Path(*candidate.parts[1:])
        if str(candidate) in {"", "."}:
            raise PogobatchError(
                f"Unsupported whole-context %files source {raw_source!r} in "
                f"{definition_path}"
            )
        full = source_root / candidate
        if not full.exists():
            raise PogobatchError(
                f"Apptainer definition context path does not exist: {candidate}"
            )
        rendered = candidate.as_posix()
        if rendered not in seen:
            seen.add(rendered)
            includes.append(rendered)

    logger.debug(
        "Apptainer definition %s uses %d explicit context path(s)",
        definition_relative.as_posix(),
        len(includes),
    )
    return includes


def _simulator_preparation_build(
    simulator_binary: str,
    source_root: Path,
    *,
    cpus_per_task: int = 1,
    memory: str = "2GiB",
    walltime: str = "01:00:00",
) -> dict[str, Any] | None:
    """Infer the normal Pogosim ``make -C DIR clean sim`` build when possible.

    The target-side preparation allocation may include verified image acquisition
    before the application compilation. Cold cluster caches can therefore take
    substantially longer than ``make sim`` alone, so use a conservative one-hour
    default and expose the bound through the cluster CLI.
    """
    source_root = source_root.expanduser().resolve()
    binary = Path(simulator_binary)
    full_binary = binary if binary.is_absolute() else source_root / binary
    try:
        relative_binary = full_binary.resolve(strict=False).relative_to(source_root)
    except ValueError:
        logger.debug(
            "Simulator binary %s is outside source root %s; no automatic preparation build",
            simulator_binary,
            source_root,
        )
        return None
    makefile = source_root / relative_binary.parent / "Makefile"
    if not makefile.is_file():
        logger.debug(
            "No Makefile beside simulator %s; no automatic preparation build",
            relative_binary,
        )
        return None
    workdir = relative_binary.parent.as_posix()
    if workdir == ".":
        argv = ["make", "clean", "sim"]
    else:
        argv = ["make", "-C", workdir, "clean", "sim"]
    return {
        "argv": argv,
        "outputs": [
            {"path": relative_binary.as_posix(), "executable": True}
        ],
        "cache_scope": "target",
        "resources": {
            "cpus_per_task": cpus_per_task,
            "memory": memory,
            "walltime": walltime,
        },
    }



_POGOSIM_FULL_IMAGE_PATTERN = re.compile(
    r"^pogosim-full[-_]v(?P<version>[0-9]+(?:\.[0-9]+){2})\.sif$"
)


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of *path* without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _infer_pogosim_prebuilt_uri(image_path: Path) -> str | None:
    """Infer the published Pogosim Library URI from a conventional SIF filename."""
    match = _POGOSIM_FULL_IMAGE_PATTERN.fullmatch(image_path.name)
    if match is None:
        return None
    return (
        "library://leo.cazenille/pogosim/pogosim-full:"
        f"v{match.group('version')}"
    )


def _plan_scheduler_kind(document: dict[str, Any]) -> str | None:
    """Return the scheduler kind from Rundra's public ``plan --json`` document."""
    plan = document.get("plan")
    if not isinstance(plan, dict):
        return None
    target = plan.get("target")
    if not isinstance(target, dict):
        return None
    scheduler = target.get("scheduler")
    if not isinstance(scheduler, dict):
        return None
    value = scheduler.get("type")
    return str(value) if value is not None else None


def _make_prebuilt_rundra_preparation(
    args: argparse.Namespace,
    *,
    generated_dir: Path,
    experiment_path: Path,
) -> tuple[Path, str] | None:
    """Prepare an explicitly supplied Pogosim SIF plus the selected application.

    A versioned ``pogosim-full`` SIF is a Pogosim toolchain/runtime image; example
    and user applications still need to be built from the staged source. Rundra's
    preparation layer is the right place to do that once per target/cache key.

    Return ``None`` when no application build can be inferred. Otherwise create a
    temporary project using a verified prebuilt image and the inferred ``make sim``
    command, and rewrite the generated experiment to reference the prepared image
    by its logical filename.
    """
    if args.source_root is None or args.container_image is None:
        return None

    source_root = Path(args.source_root).expanduser().resolve()
    inferred_build = _simulator_preparation_build(
        args.simulator_binary,
        source_root,
        cpus_per_task=args.sim_build_cpus,
        memory=args.sim_build_memory,
        walltime=args.sim_build_walltime,
    )
    if inferred_build is None:
        return None

    image_path = Path(args.container_image).expanduser()
    if not image_path.is_absolute():
        image_path = (Path.cwd() / image_path).resolve()
    else:
        image_path = image_path.resolve()

    container_uri = getattr(args, "container_uri", None)
    container_sha256 = getattr(args, "container_sha256", None)

    if container_uri is None:
        container_uri = _infer_pogosim_prebuilt_uri(image_path)

    if container_sha256 is None and image_path.is_file():
        logger.info("Hashing prebuilt Pogosim image for Rundra preparation: %s", image_path)
        container_sha256 = _sha256_file(image_path)

    if container_uri is None or container_sha256 is None:
        missing: list[str] = []
        if container_uri is None:
            missing.append("--container-uri")
        if container_sha256 is None:
            missing.append("--container-sha256")
        raise PogobatchError(
            "Remote Pogosim execution needs the selected simulator to be compiled "
            "inside the container during Rundra preparation. Pogobatch inferred the "
            f"build command '{shlex.join(inferred_build['argv'])}', but immutable "
            "prebuilt-image metadata is incomplete (missing "
            + ", ".join(missing)
            + "). For published Pogosim full images, use a conventional filename "
            "such as pogosim-full_v0.10.10.sif so the Library URI can be inferred, "
            "or provide --container-uri and --container-sha256 explicitly. Use "
            "--no-prepare-simulator only if the executable is intentionally already "
            "part of the staged source/runtime."
        )

    if re.fullmatch(r"[0-9a-fA-F]{64}", str(container_sha256)) is None:
        raise PogobatchError("--container-sha256 must contain exactly 64 hexadecimal characters")

    image_name = image_path.name
    if not image_name:
        raise PogobatchError("--container-image must identify a SIF filename")

    project: dict[str, Any] = {
        "version": 5,
        "preparation": {
            "source": {"working_tree": {}},
            "image": {
                "name": image_name,
                "prebuilt": {
                    "uri": str(container_uri),
                    "sha256": str(container_sha256).lower(),
                },
            },
            "build": inferred_build,
        },
    }
    if args.profile and not args.target:
        project["profiles"] = {args.profile: {"target": args.profile}}

    project_path = generated_dir / "pogobatch_rundra_prebuilt_project.yaml"
    project_path.write_text(yaml.safe_dump(project, sort_keys=False), encoding="utf-8")

    experiment_doc = _load_yaml(experiment_path)
    experiment_doc["container"] = {
        "image": image_name,
        "gpu": bool(args.container_gpu),
    }
    experiment_path.write_text(
        yaml.safe_dump(experiment_doc, sort_keys=False), encoding="utf-8"
    )
    logger.info(
        "Using Rundra preparation to build simulator with %s inside verified image %s",
        shlex.join(inferred_build["argv"]),
        image_name,
    )
    logger.info(
        "Simulator preparation resources: cpus=%s memory=%s walltime=%s",
        inferred_build["resources"]["cpus_per_task"],
        inferred_build["resources"]["memory"],
        inferred_build["resources"]["walltime"],
    )
    logger.debug("Prebuilt image URI: %s", container_uri)
    logger.debug("Prebuilt image SHA-256: %s", container_sha256)
    return project_path, image_name

def _make_automatic_rundra_preparation(
    args: argparse.Namespace,
    *,
    generated_dir: Path,
    experiment_path: Path,
) -> tuple[Path, str]:
    """Create a temporary Rundra v5 project using Pogosim's Apptainer definition."""
    if args.source_root is None:
        raise PogobatchError(
            "Automatic Apptainer preparation requires a Rundra --source-root"
        )
    source_root = Path(args.source_root).expanduser().resolve()
    requested_definition = getattr(args, "apptainer_def", None)
    if requested_definition is None:
        definition_path = source_root / "pogosim-apptainer.def"
    else:
        definition_path = Path(requested_definition).expanduser()
        if not definition_path.is_absolute():
            definition_path = source_root / definition_path
    definition_path = definition_path.resolve()
    if not definition_path.is_file():
        raise PogobatchError(
            "Rundra requires a container for this target, but no usable Pogosim "
            f"Apptainer definition was found at {definition_path}. Provide "
            "--container-image, --apptainer-def, or add pogosim-apptainer.def to "
            "the source checkout."
        )
    try:
        definition_relative = definition_path.relative_to(source_root).as_posix()
    except ValueError as exc:
        raise PogobatchError(
            f"--apptainer-def must be inside source root {source_root}: {definition_path}"
        ) from exc

    context = _apptainer_definition_context(definition_path, source_root)
    image_name = (
        "pogosim.sif"
        if definition_path.name == "pogosim-apptainer.def"
        else f"{definition_path.stem}.sif"
    )
    project: dict[str, Any] = {
        "version": 5,
        "preparation": {
            "source": {"working_tree": {}},
            "image": {
                "name": image_name,
                "definition": {
                    "path": definition_relative,
                    "resources": {
                        "cpus_per_task": args.image_build_cpus,
                        "memory": args.image_build_memory,
                        "walltime": args.image_build_walltime,
                    },
                    "context": {"include": context},
                },
            },
        },
    }
    inferred_build = _simulator_preparation_build(
        args.simulator_binary,
        source_root,
        cpus_per_task=args.sim_build_cpus,
        memory=args.sim_build_memory,
        walltime=args.sim_build_walltime,
    )
    if inferred_build is not None:
        project["preparation"]["build"] = inferred_build
        logger.debug(
            "Automatic Rundra preparation will build simulator with: %s",
            shlex.join(inferred_build["argv"]),
        )
    else:
        logger.warning(
            "Automatic container preparation could not infer a target-side build "
            "for simulator %s. The executable must therefore already be present in "
            "the staged source tree.",
            args.simulator_binary,
        )

    # If the user selected a profile name without an existing project file, make
    # that profile explicit in this temporary project. This works with both the
    # newest target-as-profile Rundra behavior and older strict profile resolution.
    if args.profile and not args.target:
        project["profiles"] = {args.profile: {"target": args.profile}}

    project_path = generated_dir / "pogobatch_rundra_project.yaml"
    project_path.write_text(yaml.safe_dump(project, sort_keys=False), encoding="utf-8")

    experiment_doc = _load_yaml(experiment_path)
    experiment_doc["container"] = {
        "image": image_name,
        "gpu": bool(args.container_gpu),
    }
    experiment_path.write_text(
        yaml.safe_dump(experiment_doc, sort_keys=False), encoding="utf-8"
    )
    logger.info(
        "Using automatic Pogosim Apptainer preparation from %s (image %s)",
        definition_relative,
        image_name,
    )
    return project_path, image_name


_RUNDRA_HELP_CACHE: dict[str, str] = {}


def _rundr_help_text(command: str) -> str:
    """Return help text from the installed Rundra command, cached per subcommand.

    Rundra is evolving quickly and recent documentation snapshots have not always
    matched the parser shipped by an installed binary. Pogobatch therefore treats
    ``rundr help COMMAND`` as the runtime source of truth for optional flags.
    """
    cached = _RUNDRA_HELP_CACHE.get(command)
    if cached is not None:
        return cached
    try:
        completed = subprocess.run(
            ["rundr", "help", command],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise PogobatchError(
            "Could not find 'rundr'. Install/configure Rundra before using cluster mode."
        ) from exc
    text = (completed.stdout or "") + "\n" + (completed.stderr or "")
    if completed.returncode != 0 or not text.strip():
        logger.debug(
            "Could not inspect 'rundr help %s' (exit %d); optional flags will be omitted",
            command,
            completed.returncode,
        )
        text = ""
    _RUNDRA_HELP_CACHE[command] = text
    return text


def _rundr_supports_option(command: str, option: str) -> bool:
    """Whether the installed Rundra subcommand advertises *option*."""
    return option in _rundr_help_text(command)


def _run_rundr_json(
    argv: list[str],
    *,
    stream_stderr: bool = False,
    progress: bool = False,
    verbose: bool = False,
) -> dict[str, Any]:
    command = ["rundr", *argv]
    subcommand = argv[0] if argv else ""
    if progress and _rundr_supports_option(subcommand, "--progress"):
        command.append("--progress")
    if verbose and _rundr_supports_option(subcommand, "--verbose"):
        command.append("--verbose")
    command.append("--json")
    logger.debug("Executing Rundra: %s", shlex.join(command))
    try:
        if stream_stderr:
            completed = subprocess.run(
                command,
                check=False,
                stdout=subprocess.PIPE,
                text=True,
            )
        else:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
            )
    except FileNotFoundError as exc:
        raise PogobatchError(
            "Could not find 'rundr'. Install/configure Rundra before using cluster mode."
        ) from exc

    stdout = completed.stdout.strip()
    if not stdout:
        stderr = (
            ""
            if stream_stderr
            else (completed.stderr or "").strip()
        )
        detail = stderr or f"exit status {completed.returncode}"
        raise PogobatchError(f"Rundra returned no JSON output: {detail}")
    try:
        document = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise PogobatchError(
            f"Rundra returned invalid JSON (exit {completed.returncode}): {stdout}"
        ) from exc

    # Rundra's machine interface is explicitly versioned per operation.  Do
    # not scrape human output or accept an unrelated JSON document silently.
    expected_operation = argv[0] if argv else None
    operation = document.get("operation")
    format_version = document.get("format_version")
    if expected_operation is not None and operation != expected_operation:
        raise PogobatchError(
            f"Unexpected Rundra JSON operation {operation!r}; "
            f"expected {expected_operation!r}"
        )
    if not isinstance(format_version, int) or format_version < 1:
        raise PogobatchError(
            f"Rundra returned an invalid/missing format_version: {format_version!r}"
        )

    if not document.get("ok", False):
        error = document.get("error", {})
        code = str(error.get("code", "RUNDRA_ERROR"))
        message = str(error.get("message", "Rundra operation failed"))
        details = error.get("details")
        raise RundraOperationError(
            code,
            message,
            details,
            operation=str(operation) if operation is not None else expected_operation,
        )
    # Rundra returns exit status 2 for a successfully recorded Run whose
    # scientific execution ended FAILED/CANCELLED.  Preserve the JSON so the
    # caller can decide whether --allow-partial applies.
    if completed.returncode not in {0, 2}:
        raise PogobatchError(
            f"Rundra command failed with status {completed.returncode}: {stdout}"
        )
    return document


def _rundr_run_summary(document: dict[str, Any]) -> tuple[str | None, str | None, str | None]:
    """Extract Run ID, execution state, and retrieval state from Rundra JSON."""
    run_payload = document.get("run", {})
    run_id = run_payload.get("run_id") if isinstance(run_payload, dict) else None
    if run_id is None:
        run_id = _deep_find_key(document, "run_id")
    state = run_payload.get("state") if isinstance(run_payload, dict) else None
    if state is None:
        state = _deep_find_key(document, "state")
    retrieval_state = (
        run_payload.get("retrieval_state") if isinstance(run_payload, dict) else None
    )
    if retrieval_state is None:
        retrieval_state = _deep_find_key(document, "retrieval_state")
    return (
        str(run_id) if run_id is not None else None,
        str(state) if state is not None else None,
        str(retrieval_state) if retrieval_state is not None else None,
    )



def _rundr_status_summary(
    document: dict[str, Any],
) -> tuple[str | None, str | None, str | None]:
    """Extract Run ID, execution state, and retrieval state from status JSON."""
    payload = document.get("status")
    if not isinstance(payload, dict):
        payload = {}
    run_id = payload.get("run_id")
    state = payload.get("state")
    retrieval_state = payload.get("retrieval_state")
    return (
        str(run_id) if run_id is not None else None,
        str(state) if state is not None else None,
        str(retrieval_state) if retrieval_state is not None else None,
    )

def _deep_find_key(node: Any, key: str) -> Any | None:
    if isinstance(node, dict):
        if key in node:
            return node[key]
        for value in node.values():
            found = _deep_find_key(value, key)
            if found is not None:
                return found
    elif isinstance(node, list):
        for value in node:
            found = _deep_find_key(value, key)
            if found is not None:
                return found
    return None


def _rundr_shared_launch_args(args: argparse.Namespace) -> list[str]:
    """Arguments accepted by both current ``rundr plan`` and ``rundr run``."""
    result: list[str] = []
    if args.profile:
        result.extend(["--profile", args.profile])
    if getattr(args, "target", None):
        result.extend(["--target", args.target])
    if args.project_file:
        result.extend(["--project-file", str(args.project_file)])
    if getattr(args, "targets_file", None):
        result.extend(["--targets-file", str(args.targets_file)])
    if args.source_root:
        result.extend(["--source-root", str(args.source_root)])
    if args.workers is not None:
        result.extend(["--workers", str(args.workers)])
    if args.task_slots_per_worker is not None:
        result.extend(
            ["--task-slots-per-worker", str(args.task_slots_per_worker)]
        )
    if args.prepare_location:
        result.extend(["--prepare-location", args.prepare_location])
    if args.offline:
        result.append("--offline")
    if args.rebuild:
        result.append("--rebuild")
    if args.rebuild_image:
        result.append("--rebuild-image")
    return result


def _rundr_plan_launch_args(args: argparse.Namespace) -> list[str]:
    # Current Rundra's ``plan`` parser intentionally does not accept the RunRecord
    # store or submission-confirmation arguments. Keep this builder separate from
    # ``run`` so Pogobatch tracks the real CLI contract rather than assuming the
    # two commands expose identical option sets.
    return _rundr_shared_launch_args(args)


def _rundr_run_launch_args(args: argparse.Namespace) -> list[str]:
    result = _rundr_shared_launch_args(args)
    if args.data_dir:
        result.extend(["--data-dir", str(args.data_dir)])
    if args.confirm_tasks is not None:
        result.extend(["--confirm-tasks", str(args.confirm_tasks)])
    return result


def _rundr_store_args(args: argparse.Namespace) -> list[str]:
    return ["--data-dir", str(args.data_dir)] if args.data_dir else []





def _tail_text(value: Any, limit: int = 6000) -> str:
    text = "" if value is None else str(value)
    if len(text) <= limit:
        return text
    return "... [truncated] ...\n" + text[-limit:]


def _rundr_failed_run_diagnostics(
    args: argparse.Namespace,
    run_id: str,
    *,
    max_failed_tasks: int = 3,
) -> tuple[str, dict[str, Any]]:
    """Collect bounded, public Rundra diagnostics for one failed Run."""
    diagnostics: dict[str, Any] = {
        "run_id": run_id,
        "failed_tasks": [],
        "preparation": None,
    }
    lines: list[str] = []

    # A preparation failure may prevent scientific Tasks from ever starting.
    try:
        status_document = _run_rundr_json(
            ["status", run_id, *_rundr_store_args(args)]
        )
        status = status_document.get("status")
        if not isinstance(status, dict):
            status = _deep_find_key(status_document, "status")
        preparation = status.get("preparation") if isinstance(status, dict) else None
        if isinstance(preparation, dict):
            preparation_state = str(preparation.get("state") or "").upper()
            diagnostics["preparation"] = dict(preparation)
            if preparation_state in {"FAILED", "CANCELLED"}:
                try:
                    log_document = _run_rundr_json(
                        ["logs", run_id, "--preparation", *_rundr_store_args(args)]
                    )
                    payload = log_document.get("preparation_logs", {})
                    stderr = _tail_text(payload.get("stderr") if isinstance(payload, dict) else "")
                    stdout = _tail_text(payload.get("stdout") if isinstance(payload, dict) else "")
                    diagnostics["preparation_logs"] = {
                        "stderr": stderr,
                        "stdout": stdout,
                    }
                    lines.append(f"Preparation state: {preparation_state}")
                    if stdout.strip():
                        lines.append("Preparation stdout:\n" + stdout.rstrip())
                    if stderr.strip():
                        lines.append("Preparation stderr:\n" + stderr.rstrip())
                except PogobatchError as exc:
                    lines.append(f"Could not read preparation logs: {exc}")
    except PogobatchError as exc:
        lines.append(f"Could not read Rundra status: {exc}")

    # Page through Task state only until a small diagnostic sample is found.
    failed: list[dict[str, Any]] = []
    offset = 0
    page_size = 1000
    total: int | None = None
    while len(failed) < max_failed_tasks and (total is None or offset < total):
        try:
            task_document = _run_rundr_json(
                [
                    "tasks",
                    run_id,
                    "--offset",
                    str(offset),
                    "--limit",
                    str(page_size),
                    *_rundr_store_args(args),
                ]
            )
        except PogobatchError as exc:
            lines.append(f"Could not page Rundra Tasks: {exc}")
            break
        payload = task_document.get("tasks")
        if not isinstance(payload, dict):
            break
        raw_total = payload.get("total")
        if isinstance(raw_total, int):
            total = raw_total
        items = payload.get("items")
        if not isinstance(items, list) or not items:
            break
        for item in items:
            if not isinstance(item, dict):
                continue
            state = str(item.get("state") or "").upper()
            if state in {"FAILED", "CANCELLED"}:
                failed.append(dict(item))
                if len(failed) >= max_failed_tasks:
                    break
        offset += len(items)

    diagnostics["failed_tasks"] = failed
    if failed:
        lines.append(
            "Failed Task sample: "
            + ", ".join(
                f"{item.get('task_id')} (seed={item.get('seed')}, exit={item.get('exit_code')})"
                for item in failed
            )
        )

    task_logs: list[dict[str, Any]] = []
    for item in failed:
        task_id = item.get("task_id")
        if task_id is None:
            continue
        try:
            log_document = _run_rundr_json(
                ["logs", run_id, "--task", str(task_id), *_rundr_store_args(args)]
            )
            payload = log_document.get("logs", {})
            stderr = _tail_text(payload.get("stderr") if isinstance(payload, dict) else "")
            stdout = _tail_text(payload.get("stdout") if isinstance(payload, dict) else "")
            entry = {
                "task_id": str(task_id),
                "stderr": stderr,
                "stdout": stdout,
            }
            task_logs.append(entry)
            if stderr.strip():
                lines.append(f"Task {task_id} stderr:\n{stderr.rstrip()}")
            elif stdout.strip():
                lines.append(f"Task {task_id} stdout:\n{stdout.rstrip()}")
        except PogobatchError as exc:
            lines.append(f"Could not read logs for Task {task_id}: {exc}")
    diagnostics["task_logs"] = task_logs

    lines.append(f"Inspect all Task states with: rundr tasks {shlex.quote(run_id)} --json")
    if failed:
        lines.append(
            "Inspect the first failed Task with: rundr logs "
            f"{shlex.quote(run_id)} --task {shlex.quote(str(failed[0].get('task_id')))}"
        )
    return "\n".join(lines), diagnostics

def _available_cpu_count() -> int:
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def _write_private_local_rundra_target(
    base_dir: Path,
    *,
    cpus_per_task: int = 1,
) -> tuple[Path, str]:
    """Create a current-schema local/native Rundra target for smoke tests.

    Current Rundra's local scheduler supports materialized worker pools but has no
    scheduler-driven requeue recovery. Its packaged local target therefore uses
    ``worker_pool.requeue_limit: 0``. Mirror that invariant here and size the
    local concurrency conservatively from this process' CPU affinity.
    """
    target_name = "pogobatch-local"
    target_file = base_dir / "pogobatch_local_targets.yaml"
    workspace = Path.home() / ".local" / "share" / "rundra" / "workspaces"
    task_cpus = max(1, int(cpus_per_task))
    task_slots = max(1, _available_cpu_count() // task_cpus)

    target_doc = {
        "version": 6,
        "targets": {
            target_name: {
                "transport": {"type": "local"},
                "scheduler": {"type": "local"},
                "staging": {"type": "local"},
                "container": {"type": "native"},
                "workspace": str(workspace),
                "execution": {
                    "hard_task_limit": 100000,
                    "confirmation_threshold": 1000,
                    "max_active_tasks": task_slots,
                    "max_concurrent_jobs": task_slots,
                    "max_array_size": 1000,
                    "output_shard_tasks": 1000,
                    "automatic_retrieval_threshold": 1000,
                    "worker_pool": {
                        "activation_threshold": 2,
                        "default_workers": 1,
                        "max_workers": task_slots,
                        "default_task_slots_per_worker": task_slots,
                        "max_task_slots_per_worker": task_slots,
                        "tasks_per_lease": 10,
                        "infrastructure_retry_limit": 0,
                        "requeue_limit": 0,
                    },
                },
            }
        },
    }
    target_file.write_text(
        yaml.safe_dump(target_doc, sort_keys=False),
        encoding="utf-8",
    )
    return target_file, target_name


def _default_user_targets_file() -> Path:
    xdg_config_home = os.environ.get("XDG_CONFIG_HOME")
    config_home = (
        Path(xdg_config_home).expanduser()
        if xdg_config_home
        else Path.home() / ".config"
    )
    return config_home / "rundra" / "targets.yaml"


def _selected_project_target(args: argparse.Namespace) -> str | None:
    """Best-effort resolution of profile -> target from a Rundra project file."""
    if getattr(args, "target", None):
        return str(args.target)
    project_file = getattr(args, "project_file", None)
    if project_file is None or not Path(project_file).exists():
        return None
    try:
        project_doc = _load_yaml(Path(project_file))
    except Exception:  # noqa: BLE001 - diagnostics must not mask the Rundra error
        return None

    profile_name = getattr(args, "profile", None) or project_doc.get("default_profile")
    if profile_name:
        profiles = project_doc.get("profiles", {})
        if isinstance(profiles, dict):
            profile = profiles.get(profile_name)
            if isinstance(profile, dict) and profile.get("target"):
                return str(profile["target"])

    defaults = project_doc.get("defaults")
    if isinstance(defaults, dict) and defaults.get("target"):
        return str(defaults["target"])
    if project_doc.get("target"):
        return str(project_doc["target"])
    return None


def _normalise_local_target_requeue_policy(
    args: argparse.Namespace,
    *,
    generated_dir: Path,
) -> tuple[Path, str, int] | None:
    """Create a temporary requeue-safe copy of a selected *local* target.

    Returns ``(temporary_targets_file, target_name, old_requeue_limit)`` when a
    genuine local scheduler target has a non-zero worker-pool requeue limit. The
    user's real target file is never modified. Remote scheduler targets are never
    rewritten.
    """
    target_name = _selected_project_target(args)
    if not target_name:
        return None

    source_targets = (
        Path(args.targets_file)
        if getattr(args, "targets_file", None) is not None
        else _default_user_targets_file()
    )
    if not source_targets.exists():
        return None

    try:
        targets_doc = _load_yaml(source_targets)
    except Exception:  # noqa: BLE001
        return None
    targets = targets_doc.get("targets", {})
    if not isinstance(targets, dict):
        return None
    target = targets.get(target_name)
    if not isinstance(target, dict):
        return None
    scheduler = target.get("scheduler", {})
    scheduler_type = scheduler.get("type") if isinstance(scheduler, dict) else None
    if str(scheduler_type).lower() != "local":
        return None

    execution = target.get("execution")
    if not isinstance(execution, dict):
        return None
    worker_pool = execution.get("worker_pool")
    if not isinstance(worker_pool, dict):
        return None
    try:
        old_limit = int(worker_pool.get("requeue_limit", 0))
    except (TypeError, ValueError):
        return None
    if old_limit == 0:
        return None

    rewritten = copy.deepcopy(targets_doc)
    rewritten["targets"][target_name]["execution"]["worker_pool"][
        "requeue_limit"
    ] = 0
    temp_targets = generated_dir / "pogobatch_local_requeue0_targets.yaml"
    temp_targets.write_text(
        yaml.safe_dump(rewritten, sort_keys=False),
        encoding="utf-8",
    )
    return temp_targets, target_name, old_limit


def _rundra_plan_diagnostic(exc: RundraOperationError, args: argparse.Namespace) -> PogobatchError:
    """Turn common Rundra planning failures into actionable Pogobatch messages."""
    if exc.code == "CONTAINER_REQUIRED":
        return PogobatchError(
            "Rundra selected a remote target that requires an experiment container. "
            "For generated Pogosim campaigns, leave automatic Apptainer preparation "
            "enabled (default) with pogosim-apptainer.def in the source root, or use "
            "--apptainer-def / --container-image. If an explicit --project-file is "
            "used, declare its preparation image and the matching experiment container. "
            f"Underlying error: {exc}"
        )
    if exc.code == "UNSUPPORTED_SCHEDULER_RECOVERY":
        requeue_limit = None
        if isinstance(exc.details, dict):
            requeue_limit = exc.details.get("requeue_limit")
        limit_note = (
            f" (resolved requeue_limit={requeue_limit})"
            if requeue_limit is not None
            else ""
        )
        return PogobatchError(
            "Rundra preflight rejected worker-pool recovery"
            f"{limit_note}. A genuine current Rundra local scheduler supports "
            "materialized worker pools but does not support scheduler-driven requeue "
            "recovery; its packaged local target therefore uses "
            "execution.worker_pool.requeue_limit: 0. Current `rundr run` does not "
            "expose a CLI execution-strategy override, so Pogobatch will not pass a "
            "plan-only flag to it. Set requeue_limit to 0 for the selected local "
            "target, or use --rundra-local for an isolated current-schema local "
            "target. No Rundra Run was created."
        )
    return PogobatchError(f"Rundra preflight failed: {exc}")


def _log_rundra_plan_summary(document: dict[str, Any]) -> None:
    """Log useful plan facts without depending on one plan-schema version."""
    if not logger.isEnabledFor(logging.DEBUG):
        return
    interesting = {
        "target": _deep_find_key(document, "target_name")
        or _deep_find_key(document, "target"),
        "strategy": _deep_find_key(document, "execution_strategy")
        or _deep_find_key(document, "strategy"),
        "task_count": _deep_find_key(document, "task_count"),
        "workers": _deep_find_key(document, "worker_count")
        or _deep_find_key(document, "workers"),
        "slots_per_worker": _deep_find_key(document, "task_slots_per_worker"),
    }
    rendered = ", ".join(
        f"{key}={value}" for key, value in interesting.items() if value is not None
    )
    if rendered:
        logger.debug("Rundra preflight plan: %s", rendered)


# ---------------------------------------------------------------------------
# Command implementations
# ---------------------------------------------------------------------------


def command_plan(args: argparse.Namespace) -> int:
    source = _load_yaml(args.config)
    seeds = _resolve_seeds(source, args.runs, args.seeds, args.seed)
    combinations = _filter_combinations(build_combinations(source), args.only_output)
    task_count = len(combinations) * len(seeds)
    outputs = sorted({combination.result_filename for combination in combinations})

    document = {
        "operation": "plan",
        "ok": True,
        "config": str(args.config),
        "combinations": len(combinations),
        "seeds": list(seeds),
        "task_count": task_count,
        "outputs": outputs,
        "tasks": [
            {
                "combination": combination.ordinal,
                "seed": seed,
                "result_filename": combination.result_filename,
                "config_hash": combination.config_hash,
            }
            for combination in combinations
            for seed in seeds
        ]
        if args.show_tasks
        else None,
    }
    if args.json:
        print(json.dumps(document, sort_keys=True))
    else:
        print(
            f"Combinations: {len(combinations)}\n"
            f"Seeds: {len(seeds)} ({seeds[0]}..{seeds[-1]})\n"
            f"Tasks: {task_count}\n"
            f"Outputs: {len(outputs)}"
        )
        for output in outputs:
            print(f"  {output}")
        if args.show_tasks:
            for task in document["tasks"] or []:
                print(
                    "  combo={combination} seed={seed} -> {result_filename}".format(
                        **task
                    )
                )
    return 0


def command_task(args: argparse.Namespace) -> int:
    seed = args.seed if args.seed is not None else _random_seed()
    if args.seed is None:
        logger.debug("No --seed specified; selected random seed %d", seed)

    try:
        manifest = execute_task(
            args.config,
            args.simulator_binary,
            seed,
            args.output_dir,
            gui=args.gui,
            logical_run=args.run_index,
            retry_attempt=args.retry_attempt,
            simulator_output=_simulator_output_mode(args),
        )
    except subprocess.CalledProcessError as exc:
        raise PogobatchError(
            f"Simulator exited with status {exc.returncode}"
        ) from exc

    if args.json:
        print(
            json.dumps(
                {
                    "operation": "task",
                    "ok": True,
                    "manifest": str(manifest),
                    "seed": seed,
                },
                sort_keys=True,
            )
        )
    elif not args.quiet:
        logger.info("Task completed: %s", manifest)
    return 0


def command_merge(args: argparse.Namespace) -> int:
    artifacts = discover_task_artifacts(args.inputs)
    outputs = merge_task_artifacts(
        artifacts,
        args.output_dir,
        configuration_path=args.config,
        append=args.append,
    )
    document = {
        "operation": "merge",
        "ok": True,
        "task_count": len(artifacts),
        "outputs": [str(path) for path in outputs],
    }
    if args.json:
        print(json.dumps(document, sort_keys=True))
    else:
        print(f"Merged {len(artifacts)} task shard(s) into {len(outputs)} file(s):")
        for output in outputs:
            print(f"  {output}")
    return 0


def command_run(args: argparse.Namespace) -> int:
    source = _load_yaml(args.config)
    seeds = _resolve_seeds(source, args.runs, args.seeds, args.seed)
    combinations = _filter_combinations(build_combinations(source), args.only_output)

    args.temp_base.mkdir(parents=True, exist_ok=True)
    campaign_dir = Path(
        tempfile.mkdtemp(prefix="pogobatch_", dir=str(args.temp_base))
    )
    specs = [
        LocalTaskSpec(
            combination_ordinal=combination.ordinal,
            config=combination.config,
            result_filename=combination.result_filename,
            extra_columns=combination.extra_columns,
            config_hash=combination.config_hash,
            logical_run=logical_run,
            base_seed=seed,
            all_seeds=seeds,
            simulator_binary=args.simulator_binary,
            campaign_dir=str(campaign_dir),
            gui=args.gui,
            max_retries=args.retries,
            retry_new_seed=args.retry_new_seed,
            simulator_output=_simulator_output_mode(args),
        )
        for combination in combinations
        for logical_run, seed in enumerate(seeds)
    ]

    jobs = args.jobs if args.jobs > 0 else min(len(specs), len(seeds))
    jobs = max(1, jobs)
    logger.info(
        "Running %d local task(s): %d combination(s) x %d seed(s), jobs=%d",
        len(specs),
        len(combinations),
        len(seeds),
        jobs,
    )

    try:
        results = _execute_local_specs(
            specs,
            args.backend,
            jobs,
            progress=args.progress,
        )
        failures = [result for result in results if not result.get("ok")]
        if failures:
            failure_text = "; ".join(
                "combo {combination_ordinal} run {logical_run}: {error}".format(
                    **failure
                )
                for failure in failures
            )
            raise PogobatchError(
                f"{len(failures)} local task(s) failed after retries: {failure_text}"
            )

        artifacts = discover_task_artifacts([campaign_dir])
        outputs = merge_task_artifacts(
            artifacts,
            args.output_dir,
            configuration_path=args.config,
            append=False,
        )
        _atomic_write_json(
            args.output_dir / RUN_MANIFEST_NAME,
            {
                "schema_version": 1,
                "executor": "local",
                "task_count": len(artifacts),
                "combinations": len(combinations),
                "seeds": list(seeds),
                "outputs": [str(path) for path in outputs],
            },
        )
    except Exception:
        logger.error(
            "Local campaign failed; temporary task data retained at %s", campaign_dir
        )
        raise
    else:
        if not args.keep_temp:
            shutil.rmtree(campaign_dir, ignore_errors=True)
        else:
            logger.info("Keeping local task shards at %s", campaign_dir)

    if args.json:
        print(
            json.dumps(
                {
                    "operation": "run",
                    "ok": True,
                    "task_count": len(specs),
                    "outputs": [str(path) for path in outputs],
                    "temp_dir": str(campaign_dir) if args.keep_temp else None,
                },
                sort_keys=True,
            )
        )
    elif not args.quiet:
        print(f"Completed {len(specs)} local task(s).")
        for output in outputs:
            print(f"  {output}")
    return 0


def command_cluster(args: argparse.Namespace) -> int:
    """Run one compact multi-Task Rundra campaign synchronously and merge its outputs.

    Pogobatch deliberately submits the complete parameter/seed TaskSpace as ONE
    Rundra Run.  Rundra owns execution-strategy selection, scheduler capability
    validation, source sealing, preparation, and task scheduling.

    Current Rundra ``run`` performs synchronous retrieval to ``--destination``
    but does not expose the plan-only ``--retrieval``/``--execution-strategy``
    switches. Pogobatch therefore requests ``--fetch-mode copy`` when that option
    is supported by the installed CLI and otherwise relies on an explicit
    ``rundr fetch`` fallback. Ordinary Runs should contain loose
    ``pogobatch_task.json``/Feather files and can be merged immediately. If no
    Pogobatch task artifacts are discoverable (for example, because a large
    compact Run was retrieved as verified shards), Pogobatch performs one
    idempotent ``rundr fetch --mode copy --extract`` when those fetch options are
    supported to materialize the task files before merging.

    For generated experiments on remote Apptainer targets, a CONTAINER_REQUIRED
    preflight can be satisfied automatically from Pogosim's shipped
    ``pogosim-apptainer.def``. Pogobatch emits a temporary Rundra v5 preparation
    project using working-tree source, the definition's explicit %files context,
    and (when inferable) a target-side ``make ... sim`` application build. The
    user's target policy remains authoritative for whether definition builds are
    permitted and where they execute.

    After a fully successful run/retrieval/merge, the per-Run Rundra workspace
    is purged by default to avoid accumulating source snapshots and raw outputs.
    The client RunRecord and merged Pogobatch results remain.  Use
    ``--keep-rundra-workspace`` while debugging or when raw workspace retention
    is desired.
    """
    generated_dir: Path | None = None
    generated_experiment: Path | None = None
    generated_config: Path | None = None
    merge_config = args.config
    task_count: int | None = None

    if getattr(args, "image_build_cpus", 1) <= 0:
        raise PogobatchError("--image-build-cpus must be greater than zero")
    if getattr(args, "sim_build_cpus", 1) <= 0:
        raise PogobatchError("--sim-build-cpus must be greater than zero")

    if args.experiment is None:
        if args.config is None or args.simulator_binary is None:
            raise PogobatchError(
                "Generated cluster mode requires --config and --simulator-binary"
            )
        source = _load_yaml(args.config)
        _, pogobatch_settings = _split_pogobatch_config(source, warn_legacy=True)
        rundra_sync_exclude = _pogobatch_rundra_sync_exclude(pogobatch_settings)
        resolved_seeds = _resolve_seeds(source, args.runs, args.seeds, args.seed)

        # The normal cluster path keeps the scientific sweep compact and lets
        # Rundra perform the native deterministic expansion. This avoids writing
        # one complete configuration copy per Cartesian-product combination.
        # --only-output is a general correlated filter that cannot always be
        # represented as independent sweep factors, so retain the legacy resolved
        # wrapper only for that explicit compatibility case.
        if args.only_output:
            combinations = _filter_combinations(
                build_combinations(source), args.only_output
            )
            combination_count = len(combinations)
            rundra_config_document = make_rundra_wrapper_config(combinations)
            logger.warning(
                "--only-output requires materializing the selected combination set; "
                "compact Rundra sweep generation is disabled for this campaign"
            )
        else:
            combination_count = count_combinations(source)
            rundra_config_document = make_rundra_sweep_config(source)

        task_count = combination_count * len(resolved_seeds)

        args.temp_base.mkdir(parents=True, exist_ok=True)
        generated_dir = Path(
            tempfile.mkdtemp(prefix="pogobatch_rundra_", dir=str(args.temp_base))
        )
        generated_config = generated_dir / "pogobatch_rundra_config.yaml"
        generated_experiment = generated_dir / "pogobatch_rundra_experiment.yaml"
        generated_config.write_text(
            yaml.safe_dump(rundra_config_document, sort_keys=False),
            encoding="utf-8",
        )
        logger.debug(
            "Generated compact Rundra config: %s bytes for %d combination(s)",
            generated_config.stat().st_size,
            combination_count,
        )
        task_launcher = shlex.split(args.task_launcher)
        if not task_launcher:
            raise PogobatchError("--task-launcher cannot be empty")
        generated_experiment.write_text(
            yaml.safe_dump(
                make_rundra_experiment(
                    name=args.experiment_name
                    or f"pogosim-{args.config.stem}",
                    simulator_binary=args.simulator_binary,
                    task_launcher=task_launcher,
                    cpus_per_task=args.cpus_per_task,
                    gpus_per_task=args.gpus_per_task,
                    memory=args.memory,
                    walltime=args.walltime,
                    container_image=args.container_image,
                    container_gpu=args.container_gpu,
                    sync_exclude=rundra_sync_exclude,
                ),
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        experiment = generated_experiment
        rundra_config = generated_config
        # Pogobatch currently expands explicit seed sets to a contiguous range.
        # Rundra's START:STOP syntax is inclusive.
        rundra_seed_args = [
            "--seeds",
            f"{resolved_seeds[0]}:{resolved_seeds[-1]}",
        ]
    else:
        experiment = args.experiment
        rundra_config = args.config
        rundra_seed_args: list[str] = []
        if args.seed is not None:
            if args.seed < 0:
                raise PogobatchError("--seed must be non-negative")
            rundra_seed_args = ["--seed", str(args.seed)]
        elif args.seeds is not None:
            rundra_seed_args = ["--seeds", args.seeds]
        elif args.runs is not None:
            if args.runs <= 0:
                raise PogobatchError("--runs must be greater than zero")
            rundra_seed_args = ["--seeds", f"0:{args.runs - 1}"]

    # ``--rundra-local`` is an explicit zero-ambiguity integration-test mode.
    # It intentionally bypasses adjacent project profiles and user target routing.
    # A profile merely named "local" can legally point at OpenPBS/Slurm/etc.
    if args.rundra_local:
        conflicts = []
        for option_name in ("profile", "target", "project_file", "targets_file"):
            if getattr(args, option_name, None) is not None:
                conflicts.append("--" + option_name.replace("_", "-"))
        if conflicts:
            raise PogobatchError(
                "--rundra-local cannot be combined with " + ", ".join(conflicts)
            )
        if generated_dir is None:
            args.temp_base.mkdir(parents=True, exist_ok=True)
            generated_dir = Path(
                tempfile.mkdtemp(prefix="pogobatch_rundra_", dir=str(args.temp_base))
            )
        local_targets_file, local_target_name = _write_private_local_rundra_target(
            generated_dir,
            cpus_per_task=args.cpus_per_task,
        )
        args.target = local_target_name
        args.targets_file = local_targets_file
        if args.source_root is None:
            args.source_root = Path.cwd()
        logger.debug(
            "Using private Rundra local/native target %s from %s",
            local_target_name,
            local_targets_file,
        )
    else:
        # ``rundra.yaml`` is optional in current Rundra.  Profiles only exist in
        # that project file, however, whereas targets are independently available
        # from the user/built-in targets configuration.  For generated Pogobatch
        # experiments, look for a project file in the source checkout (cwd); for
        # an explicitly supplied experiment, follow Rundra's adjacent-project
        # convention.
        if args.project_file is not None:
            args.project_file = Path(args.project_file)
            if not args.project_file.exists():
                raise PogobatchError(
                    f"Rundra project file does not exist: {args.project_file}"
                )
        else:
            if generated_experiment is not None:
                adjacent_project = Path.cwd() / "rundra.yaml"
            else:
                adjacent_project = Path(experiment).resolve().parent / "rundra.yaml"
            if adjacent_project.exists():
                args.project_file = adjacent_project

        # New Rundra releases accept target names through --profile even without
        # a project file. Preserve the selector exactly as requested and let the
        # installed Rundra resolve it first. If an older release returns
        # PROFILE_NOT_FOUND, the preflight recovery path below retries the same
        # name as --target without creating a Run.

    # A generated experiment lives in a temporary directory. Without an
    # adjacent Rundra project or an explicit source root, Rundra would otherwise
    # infer that temporary directory instead of the Pogosim checkout.
    if (
        generated_experiment is not None
        and args.project_file is None
        and args.source_root is None
    ):
        args.source_root = Path.cwd()

    args.temp_base.mkdir(parents=True, exist_ok=True)
    fetch_destination = args.fetch_destination
    if fetch_destination is None:
        fetch_destination = (
            args.temp_base / f"pogobatch_fetch_{uuid.uuid4().hex}"
        )
    fetched_path = Path(fetch_destination)

    # ONE Rundra Run regardless of the number of Pogobatch Tasks. ``run``
    # synchronously retrieves the experiment's declared outputs. Build argv from
    # the installed CLI surface so recovery steps can adjust profile/target or
    # preparation without carrying stale command-line options forward.
    def build_rundra_launch_argv() -> tuple[list[str], list[str], list[str]]:
        plan = ["plan", str(experiment)]
        run = ["run", str(experiment)]
        submit = ["submit", str(experiment)]
        if rundra_config is not None:
            plan.extend(["--config", str(rundra_config)])
            run.extend(["--config", str(rundra_config)])
            submit.extend(["--config", str(rundra_config)])
        plan.extend(rundra_seed_args)
        run.extend(rundra_seed_args)
        submit.extend(rundra_seed_args)
        plan.extend(_rundr_plan_launch_args(args))
        run.extend(_rundr_run_launch_args(args))
        submit.extend(_rundr_run_launch_args(args))
        if _rundr_supports_option("plan", "--fetch-mode"):
            plan.extend(["--fetch-mode", "copy"])
        if _rundr_supports_option("run", "--fetch-mode"):
            run.extend(["--fetch-mode", "copy"])
        if _rundr_supports_option("submit", "--fetch-mode"):
            submit.extend(["--fetch-mode", "copy"])
        run.extend(["--destination", str(fetched_path)])
        submit.extend(["--destination", str(fetched_path)])
        return plan, run, submit

    # ``plan`` is side-effect free. Use it as the compatibility/preparation
    # boundary before any Run is created. Recovery is intentionally narrow:
    # - newest Rundra resolves target names through --profile without rundra.yaml;
    #   older releases get one PROFILE_NOT_FOUND -> --target retry;
    # - remote Apptainer targets can trigger automatic preparation from Pogosim's
    #   shipped definition file when no explicit image/project preparation exists;
    # - genuine local targets may need their legacy nonzero requeue limit normalized.
    profile_fallback_used = False
    automatic_apptainer_used = False
    automatic_prebuilt_preparation_used = False
    local_requeue_normalized = False
    while True:
        plan_argv, run_argv, submit_argv = build_rundra_launch_argv()
        try:
            plan_document = _run_rundr_json(plan_argv)

            # A prebuilt Pogosim SIF provides the toolchain/runtime, but normal
            # example/user simulator binaries are gitignored and must be compiled
            # against that runtime. For remote schedulers, synthesize the same
            # preparation build used by Rundra's checked Pogosim example. Do this
            # only after a side-effect-free plan has resolved the actual scheduler.
            if (
                generated_experiment is not None
                and generated_dir is not None
                and args.container_image is not None
                and args.project_file is None
                and getattr(args, "prepare_simulator", True)
                and not automatic_prebuilt_preparation_used
                and _plan_scheduler_kind(plan_document) not in {None, "local"}
            ):
                prepared = _make_prebuilt_rundra_preparation(
                    args,
                    generated_dir=generated_dir,
                    experiment_path=generated_experiment,
                )
                if prepared is not None:
                    auto_project, image_name = prepared
                    args.project_file = auto_project
                    automatic_prebuilt_preparation_used = True
                    setattr(args, "_automatic_prebuilt_image", image_name)
                    # Re-plan with the generated preparation recipe so the actual
                    # Run cannot bypass application compilation.
                    continue
            break
        except RundraOperationError as exc:
            if (
                exc.code == "PROFILE_NOT_FOUND"
                and args.project_file is None
                and args.profile is not None
                and args.target is None
                and not profile_fallback_used
            ):
                old_profile = args.profile
                args.profile = None
                args.target = old_profile
                profile_fallback_used = True
                logger.debug(
                    "Installed Rundra did not resolve --profile %s without a project; "
                    "retrying preflight as --target %s",
                    old_profile,
                    old_profile,
                )
                continue

            if (
                exc.code == "CONTAINER_REQUIRED"
                and generated_experiment is not None
                and args.container_image is None
                and args.project_file is None
                and args.auto_apptainer
                and not automatic_apptainer_used
            ):
                if generated_dir is None:
                    raise PogobatchError(
                        "Internal error: automatic Apptainer preparation needs generated files"
                    ) from exc
                auto_project, image_name = _make_automatic_rundra_preparation(
                    args,
                    generated_dir=generated_dir,
                    experiment_path=generated_experiment,
                )
                args.project_file = auto_project
                automatic_apptainer_used = True
                setattr(args, "_automatic_apptainer_image", image_name)
                continue

            if exc.code == "UNSUPPORTED_SCHEDULER_RECOVERY" and not local_requeue_normalized:
                if generated_dir is None:
                    args.temp_base.mkdir(parents=True, exist_ok=True)
                    generated_dir = Path(
                        tempfile.mkdtemp(
                            prefix="pogobatch_rundra_", dir=str(args.temp_base)
                        )
                    )
                normalized = _normalise_local_target_requeue_policy(
                    args,
                    generated_dir=generated_dir,
                )
                if normalized is not None:
                    temp_targets, target_name, old_limit = normalized
                    logger.warning(
                        "Selected Rundra target %s uses the local scheduler with "
                        "worker_pool.requeue_limit=%d; current local worker pools "
                        "require 0. Using a temporary requeue_limit=0 target copy for "
                        "this campaign; the original targets file is unchanged.",
                        target_name,
                        old_limit,
                    )
                    args.targets_file = temp_targets
                    local_requeue_normalized = True
                    continue

            if automatic_apptainer_used:
                text = f"{exc.code} {exc.rundra_message}".lower()
                if "prepar" in text or "definition" in text or "apptainer" in text:
                    raise PogobatchError(
                        "Rundra rejected Pogobatch's automatic Apptainer preparation. "
                        "The selected target must permit definition builds through "
                        "target preparation.definition_build policy, or you can provide "
                        "an already provisioned image with --container-image / an explicit "
                        f"Rundra project preparation recipe. Underlying error: {exc}"
                    ) from exc
            raise _rundra_plan_diagnostic(exc, args) from exc

    _log_rundra_plan_summary(plan_document)

    scheduler_kind = _plan_scheduler_kind(plan_document)
    remote_async_lifecycle = scheduler_kind not in {None, "local"}

    if remote_async_lifecycle:
        # Remote schedulers have a durable scheduler owner, so use Rundra's
        # explicit asynchronous lifecycle even though `pogobatch cluster` remains
        # a synchronous human convenience command. This exposes the Run ID as soon
        # as scheduler identities are durable and makes long/cold preparation jobs
        # inspectable from another terminal while Pogobatch waits.
        logger.info("Submitting Rundra campaign...")
        submit_document = _run_rundr_json(
            submit_argv,
            stream_stderr=not args.json,
            verbose=args.verbose,
        )
        run_id, _submitted_state, retrieval_state = _rundr_run_summary(
            submit_document
        )
        if run_id is None:
            raise PogobatchError(
                f"Could not extract Rundra Run ID from submit JSON: {submit_document}"
            )
        if not args.json and not args.quiet:
            print(
                f"Rundra run {run_id} submitted; waiting for preparation and tasks...",
                file=sys.stderr,
            )
            print(
                f"Inspect concurrently with: rundr status {run_id} --json",
                file=sys.stderr,
            )
            print(
                f"Preparation logs: rundr logs {run_id} --preparation",
                file=sys.stderr,
            )

        try:
            _run_rundr_json(
                ["wait", run_id, *_rundr_store_args(args)],
                stream_stderr=not args.json,
                progress=not args.json,
                verbose=args.verbose,
            )
        except KeyboardInterrupt:
            if not args.json:
                print(
                    f"\nStopped waiting; Rundra run {run_id} remains managed by "
                    "the remote scheduler.",
                    file=sys.stderr,
                )
                print(
                    f"Resume with: rundr wait {run_id} --progress",
                    file=sys.stderr,
                )
            raise
        status_document = _run_rundr_json(
            ["status", run_id, *_rundr_store_args(args)]
        )
        status_run_id, run_state, status_retrieval_state = _rundr_status_summary(
            status_document
        )
        if status_run_id is not None and status_run_id != run_id:
            raise PogobatchError(
                f"Rundra status returned Run ID {status_run_id}, expected {run_id}"
            )
        if status_retrieval_state is not None:
            retrieval_state = status_retrieval_state
    else:
        logger.info("Running Rundra campaign synchronously...")
        run_document = _run_rundr_json(
            run_argv,
            stream_stderr=not args.json,
            progress=not args.json,
            verbose=args.verbose,
        )
        run_id, run_state, retrieval_state = _rundr_run_summary(run_document)
        if run_id is None:
            raise PogobatchError(
                f"Could not extract Rundra Run ID from run JSON: {run_document}"
            )

    # Rundra deliberately returns structured JSON for scientifically failed Runs.
    # Do not mask that failure with a later "no manifests" merge error. Surface
    # bounded framework-managed Task/preparation logs immediately. Only
    # --allow-partial proceeds to retrieval/merge of whatever succeeded.
    if run_state != "SUCCEEDED" and not args.allow_partial:
        diagnostic_text, diagnostic_document = _rundr_failed_run_diagnostics(
            args, run_id
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(
            args.output_dir / RUN_MANIFEST_NAME,
            {
                "schema_version": 1,
                "executor": "rundra",
                "rundra_run_id": run_id,
                "rundra_state": run_state,
                "rundra_retrieval_state": retrieval_state,
                "planned_task_count": task_count,
                "automatic_apptainer_preparation": automatic_apptainer_used,
                "automatic_prebuilt_preparation": automatic_prebuilt_preparation_used,
                "automatic_prebuilt_image": getattr(
                    args, "_automatic_prebuilt_image", None
                ),
                "failure_diagnostics": diagnostic_document,
                "fetched_from": str(fetched_path),
            },
        )
        retained: list[str] = [
            f"Rundra workspace retained for diagnosis (Run ID: {run_id}).",
            f"Retrieval directory retained at: {fetched_path}",
        ]
        if generated_dir is not None:
            retained.append(f"Generated Rundra files retained at: {generated_dir}")
        raise PogobatchError(
            f"Rundra run {run_id} ended in state {run_state or 'UNKNOWN'}. "
            "Pogobatch will not attempt a normal merge of a failed campaign.\n"
            + diagnostic_text
            + "\n"
            + "\n".join(retained)
        )

    # For normal materialized Runs, `rundr run` has already fetched loose files
    # and no second transfer is needed.  Large compact Runs may instead leave
    # verified shard archives/reference metadata; only then ask fetch to extract
    # individual task files.  `fetch` is documented as idempotent, so using the
    # same destination is safe.
    fetch_error: str | None = None
    fetch_document: dict[str, Any] | None = None
    used_extract_fetch = False

    # Do not call discover_task_artifacts() merely to test presence: that
    # function intentionally raises when no manifest exists.  Absence of a
    # loose task manifest is the signal that a compact/sharded retrieval may
    # need explicit extraction.
    have_loose_manifests = (
        fetched_path.is_dir()
        and any(fetched_path.rglob(TASK_MANIFEST_NAME))
    )
    artifacts: list[TaskArtifact] = (
        discover_task_artifacts([fetched_path])
        if have_loose_manifests
        else []
    )

    if not artifacts:
        if remote_async_lifecycle:
            logger.info(
                "Fetching Rundra task outputs after remote completion "
                "with 'rundr fetch --extract'."
            )
        else:
            logger.info(
                "No loose Pogobatch task artifacts found after Rundra run; "
                "materializing outputs with 'rundr fetch --extract'."
            )
        fetch_argv = [
            "fetch",
            run_id,
            "--destination",
            str(fetched_path),
        ]
        if _rundr_supports_option("fetch", "--mode"):
            fetch_argv.extend(["--mode", "copy"])
        if _rundr_supports_option("fetch", "--extract"):
            fetch_argv.append("--extract")
            used_extract_fetch = True
        fetch_argv.extend(_rundr_store_args(args))
        try:
            fetch_document = _run_rundr_json(
                fetch_argv,
                stream_stderr=not args.json,
                progress=not args.json,
                verbose=args.verbose,
            )
        except PogobatchError as exc:
            # A failed scientific Run can still have useful successful-task
            # output. Preserve any materialized files and let --allow-partial
            # decide whether a partial merge is acceptable.
            fetch_error = str(exc)
            logger.warning("Rundra fetch was not fully successful: %s", exc)

        if fetch_document is not None:
            fetch_payload = fetch_document.get("fetch")
            value = (
                fetch_payload.get("retrieval_state")
                if isinstance(fetch_payload, dict)
                else None
            )
            if value is None:
                value = _deep_find_key(fetch_document, "retrieval_state")
            if value is not None:
                retrieval_state = str(value)

        artifacts = discover_task_artifacts([fetched_path])

    merge_error: Exception | None = None
    outputs: list[Path] = []
    try:
        if not artifacts:
            metadata_path = fetched_path / "metadata" / "tasks.json"
            metadata_note = (
                f" Rundra task metadata exists at {metadata_path}; inspect its "
                "output-directory entries and task logs."
                if metadata_path.exists()
                else ""
            )
            raise PogobatchError(
                "Rundra completed retrieval, but no Pogobatch task artifacts "
                f"were found under {fetched_path}." + metadata_note
            )
        outputs = merge_task_artifacts(
            artifacts,
            args.output_dir,
            configuration_path=merge_config,
            append=False,
        )
    except Exception as exc:  # noqa: BLE001 - retain fetched data for diagnosis
        merge_error = exc

    scientific_success = run_state == "SUCCEEDED"
    complete_success = (
        scientific_success
        and fetch_error is None
        and merge_error is None
        and bool(artifacts)
    )
    partial_success = merge_error is None and bool(artifacts)

    # Avoid accumulating immutable source/output snapshots in target workspaces.
    # Rundra's purge --workspace operates on this exact terminal Run workspace;
    # the client RunRecord and our copied/merged results are separate.
    workspace_purged = False
    cleanup_error: str | None = None
    if complete_success and not args.keep_rundra_workspace:
        try:
            _run_rundr_json(
                [
                    "purge",
                    run_id,
                    "--workspace",
                    "--confirm",
                    run_id,
                    *_rundr_store_args(args),
                ]
            )
            workspace_purged = True
        except PogobatchError as exc:
            cleanup_error = str(exc)
            logger.warning(
                "Results are complete, but Rundra workspace cleanup failed: %s",
                exc,
            )

    if merge_error is None:
        _atomic_write_json(
            args.output_dir / RUN_MANIFEST_NAME,
            {
                "schema_version": 1,
                "executor": "rundra",
                "rundra_run_id": run_id,
                "rundra_state": run_state,
                "rundra_retrieval_state": retrieval_state,
                "rundra_extract_fetch_used": used_extract_fetch,
                "rundra_workspace_purged": workspace_purged,
                "rundra_cleanup_error": cleanup_error,
                "automatic_apptainer_preparation": automatic_apptainer_used,
                "automatic_apptainer_image": getattr(
                    args, "_automatic_apptainer_image", None
                ),
                "automatic_prebuilt_preparation": automatic_prebuilt_preparation_used,
                "automatic_prebuilt_image": getattr(
                    args, "_automatic_prebuilt_image", None
                ),
                "task_count": len(artifacts),
                "planned_task_count": task_count,
                "outputs": [str(path) for path in outputs],
                "fetched_from": str(fetched_path),
            },
        )

    keep_fetch = (
        args.keep_fetch
        or not complete_success
        or merge_error is not None
    )
    if not keep_fetch:
        shutil.rmtree(fetched_path, ignore_errors=True)
    elif fetched_path.exists():
        print(f"Fetched task outputs retained at: {fetched_path}", file=sys.stderr)

    if generated_dir is not None:
        if args.keep_rundra_files:
            print(
                f"Generated Rundra files retained at: {generated_dir}",
                file=sys.stderr,
            )
        else:
            shutil.rmtree(generated_dir, ignore_errors=True)

    if merge_error is not None:
        raise PogobatchError(
            f"Rundra run completed but Pogobatch merge failed: {merge_error}"
        )

    accepted = complete_success or (args.allow_partial and partial_success)
    if args.json:
        print(
            json.dumps(
                {
                    "operation": "cluster",
                    "ok": accepted,
                    "run_id": run_id,
                    "state": run_state,
                    "retrieval_state": retrieval_state,
                    "task_count": len(artifacts),
                    "planned_task_count": task_count,
                    "outputs": [str(path) for path in outputs],
                    "fetched_path": str(fetched_path) if keep_fetch else None,
                    "extract_fetch_used": used_extract_fetch,
                    "workspace_purged": workspace_purged,
                    "cleanup_error": cleanup_error,
                    "fetch_error": fetch_error,
                    "automatic_apptainer_preparation": automatic_apptainer_used,
                    "automatic_apptainer_image": getattr(
                        args, "_automatic_apptainer_image", None
                    ),
                    "automatic_prebuilt_preparation": automatic_prebuilt_preparation_used,
                    "automatic_prebuilt_image": getattr(
                        args, "_automatic_prebuilt_image", None
                    ),
                },
                sort_keys=True,
            )
        )
    elif not args.quiet:
        print(f"Rundra run {run_id}: {run_state or 'UNKNOWN'}")
        if retrieval_state is not None:
            print(f"Retrieval: {retrieval_state}")
        if used_extract_fetch:
            print("Retrieval materialization: explicit fetch --extract")
        else:
            print("Retrieval materialization: provided by rundr run")
        if workspace_purged:
            print("Rundra workspace: purged")
        elif args.keep_rundra_workspace:
            print("Rundra workspace: retained (--keep-rundra-workspace)")
        elif cleanup_error is not None:
            print("Rundra workspace: cleanup failed (see warning above)")
        elif not scientific_success:
            print("Rundra workspace: retained for failed-run diagnosis")
        for output in outputs:
            print(f"  {output}")

    return 0 if accepted else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def command_help(args: argparse.Namespace) -> int:
    """Print general Pogobatch help or help for one subcommand."""
    parser = build_parser()
    if args.topic is None:
        parser.print_help()
        return 0

    subparser_action = next(
        (action for action in parser._actions
         if isinstance(action, argparse._SubParsersAction)),
        None,
    )
    if subparser_action is None or args.topic not in subparser_action.choices:
        raise PogobatchError(f"Unknown help topic: {args.topic}")

    subparser_action.choices[args.topic].print_help()
    return 0


def _add_seed_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Use exactly this seed. If no seed/runs/range is specified and the "
            "configuration has no _rundr.seeds, choose one seed randomly."
        ),
    )
    group.add_argument(
        "-r",
        "--runs",
        type=int,
        default=None,
        help="Number of seeds/replicates, using seeds 0..N-1.",
    )
    group.add_argument(
        "--seeds",
        type=str,
        default=None,
        metavar="START:STOP",
        help="Inclusive explicit seed range. Overrides _rundr.seeds.",
    )


def _add_verbosity_arguments(
    parser: argparse.ArgumentParser,
    *,
    suppress_defaults: bool = False,
) -> None:
    default = argparse.SUPPRESS if suppress_defaults else False
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=default,
        help="Show full Pogobatch and simulator output.",
    )
    group.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        default=default,
        help="Show errors only and suppress normal completion summaries.",
    )


def _simulator_output_mode(args: argparse.Namespace) -> str:
    if getattr(args, "verbose", False):
        return "verbose"
    if getattr(args, "quiet", False):
        return "quiet"
    return "normal"


def _add_common_batch_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        required=True,
        help="Pogosim YAML configuration, possibly containing batch markers.",
    )
    _add_seed_arguments(parser)
    parser.add_argument(
        "--only-output",
        default=None,
        help="Restrict execution to combinations whose computed output matches this name.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plan, execute, distribute, and merge Pogosim batch experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Common workflows:\n"
            "  pogobatch run -c CONFIG -S SIMULATOR -r RUNS -o RESULTS\n"
            "  pogobatch cluster -c CONFIG -S SIMULATOR -r RUNS --profile PROFILE -o RESULTS\n"
            "  pogobatch merge FETCHED_CAMPAIGN -o RESULTS\n"
            "\n"
            "Batch-only YAML settings live under 'pogobatch:'. For example:\n"
            "  pogobatch.rundra.sync.exclude: [build/, large_data/]\n"
            "\n"
            "For command-specific help:\n"
            "  pogobatch help <command>\n"
            "  pogobatch <command> --help"
        ),
    )
    parser.add_argument(
        "-V", "--version", action="version", version=f"Pogobatch v{POGOBATCH_SCRIPT_VERSION} (Pogosim {__version__})"
    )
    _add_verbosity_arguments(parser)
    subparsers = parser.add_subparsers(dest="command", required=True)

    help_parser = subparsers.add_parser(
        "help", help="show general help or help for a specific subcommand"
    )
    help_parser.add_argument(
        "topic",
        nargs="?",
        choices=("help", "plan", "task", "run", "merge", "cluster"),
        help="Subcommand to describe. Omit for the general Pogobatch help.",
    )
    help_parser.set_defaults(handler=command_help)

    plan = subparsers.add_parser("plan", help="expand a batch without executing it")
    _add_verbosity_arguments(plan, suppress_defaults=True)
    _add_common_batch_arguments(plan)
    plan.add_argument("--show-tasks", action="store_true")
    plan.add_argument("--json", action="store_true")
    plan.set_defaults(handler=command_plan)

    task = subparsers.add_parser(
        "task", help="execute one configuration using batch defaults (random seed if omitted)"
    )
    _add_verbosity_arguments(task, suppress_defaults=True)
    task.add_argument("-c", "--config", type=Path, required=True)
    task.add_argument("-S", "--simulator-binary", required=True)
    task.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Explicit non-negative seed. If omitted, choose one randomly.",
    )
    task.add_argument("-o", "--output-dir", type=Path, required=True)
    task.add_argument("--run-index", type=int, default=None)
    task.add_argument("--retry-attempt", type=int, default=0)
    task.add_argument("--gui", action="store_true")
    task.add_argument("--json", action="store_true")
    task.set_defaults(handler=command_task)

    run = subparsers.add_parser("run", help="execute a complete batch locally")
    _add_verbosity_arguments(run, suppress_defaults=True)
    _add_common_batch_arguments(run)
    run.add_argument("-S", "--simulator-binary", required=True)
    run.add_argument("-o", "--output-dir", type=Path, default=Path("."))
    run.add_argument("-t", "--temp-base", type=Path, default=Path("tmp"))
    run.add_argument(
        "--backend",
        choices=("multiprocessing", "ray", "sequential"),
        default="multiprocessing",
    )
    run.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=0,
        help="Maximum concurrent local tasks; 0 defaults to the number of seeds.",
    )
    run.add_argument("--keep-temp", action="store_true")
    run.add_argument(
        "--progress",
        action="store_true",
        help="Show a tqdm progress bar for completed local tasks.",
    )
    run.add_argument("--gui", action="store_true")
    run.add_argument("-R", "--retries", type=int, default=5)
    run.add_argument(
        "--retry-same-seed",
        dest="retry_new_seed",
        action="store_false",
        help="Retry a crashing local task with the same seed instead of a replacement seed.",
    )
    run.set_defaults(retry_new_seed=True)
    run.add_argument("--json", action="store_true")
    run.set_defaults(handler=command_run)

    merge = subparsers.add_parser(
        "merge", help="merge self-describing Pogobatch task shards"
    )
    _add_verbosity_arguments(merge, suppress_defaults=True)
    merge.add_argument("inputs", type=Path, nargs="+")
    merge.add_argument("-o", "--output-dir", type=Path, default=Path("."))
    merge.add_argument(
        "-c",
        "--config",
        type=Path,
        default=None,
        help="Optional original batch config to embed as Feather schema metadata.",
    )
    merge.add_argument("--append", action="store_true")
    merge.add_argument("--json", action="store_true")
    merge.set_defaults(handler=command_merge)

    cluster = subparsers.add_parser(
        "cluster",
        help="Rundra synchronous run + Pogobatch merge",
    )
    _add_verbosity_arguments(cluster, suppress_defaults=True)
    cluster.add_argument(
        "experiment",
        type=Path,
        nargs="?",
        default=None,
        help=(
            "Existing Rundra experiment YAML. If omitted, Pogobatch generates a simple "
            "experiment from --config/--simulator-binary."
        ),
    )
    cluster.add_argument("-c", "--config", type=Path, default=None)
    cluster.add_argument("-S", "--simulator-binary", default=None)
    _add_seed_arguments(cluster)
    cluster.add_argument("--only-output", default=None)
    cluster.add_argument("-o", "--output-dir", type=Path, default=Path("."))
    cluster.add_argument("-t", "--temp-base", type=Path, default=Path("tmp"))
    cluster.add_argument(
        "--profile",
        default=None,
        help=(
            "Rundra launch profile. Current Rundra also accepts target names as "
            "profile names without rundra.yaml; Pogobatch falls back to --target "
            "only when an older installed Rundra returns PROFILE_NOT_FOUND."
        ),
    )
    cluster.add_argument(
        "--target",
        default=None,
        help=(
            "Rundra execution target. This is the preferred selector when no "
            "rundra.yaml is used. If neither --target nor --profile is supplied, "
            "Rundra may use its zero-configuration built-in local target."
        ),
    )
    cluster.add_argument(
        "--project-file",
        type=Path,
        default=None,
        help=(
            "Optional Rundra project launch file (rundra.yaml). If omitted, "
            "Pogobatch uses an adjacent file when present; otherwise Rundra's "
            "project-file-free defaults are used."
        ),
    )
    cluster.add_argument("--targets-file", type=Path, default=None)
    cluster.add_argument(
        "--rundra-local",
        action="store_true",
        help=(
            "Force an isolated local/native Rundra target for integration smoke tests. "
            "This bypasses project profiles and configured remote targets; do not combine "
            "it with --profile, --target, --project-file, or --targets-file."
        ),
    )
    cluster.add_argument("--source-root", type=Path, default=None)
    cluster.add_argument("--data-dir", type=Path, default=None)
    cluster.add_argument("--workers", type=int, default=None)
    cluster.add_argument("--task-slots-per-worker", type=int, default=None)
    cluster.add_argument("--confirm-tasks", type=int, default=None)
    cluster.add_argument(
        "--prepare-location", choices=("auto", "local", "target"), default=None
    )
    cluster.add_argument("--offline", action="store_true")
    cluster.add_argument("--rebuild", action="store_true")
    cluster.add_argument("--rebuild-image", action="store_true")
    cluster.add_argument("--fetch-destination", type=Path, default=None)
    cluster.add_argument("--keep-fetch", action="store_true")
    cluster.add_argument("--keep-rundra-files", action="store_true")
    cluster.add_argument(
        "--keep-rundra-workspace",
        action="store_true",
        help=(
            "Keep Rundra's per-Run source/output workspace after a fully successful "
            "fetch and merge. By default Pogobatch purges that exact workspace while "
            "retaining the Rundra RunRecord and merged results."
        ),
    )
    cluster.add_argument("--allow-partial", action="store_true")

    # Simple generated-experiment options. Advanced deployments should supply
    # an explicit Rundra experiment/project instead of mirroring every Rundra knob.
    cluster.add_argument("--experiment-name", default=None)
    cluster.add_argument(
        "--task-launcher",
        default="pogobatch",
        help="Command prefix available on the target for invoking this Pogobatch CLI.",
    )
    cluster.add_argument("--cpus-per-task", type=int, default=1)
    cluster.add_argument("--gpus-per-task", type=int, default=0)
    cluster.add_argument("--memory", default="1GiB")
    cluster.add_argument("--walltime", default="00:15:00")
    cluster.add_argument(
        "--container-image",
        default=None,
        help=(
            "Explicit Rundra experiment container image. This takes precedence over "
            "automatic preparation from Pogosim's Apptainer definition."
        ),
    )
    cluster.add_argument(
        "--container-uri",
        default=None,
        help=(
            "Immutable URI for a prebuilt --container-image when Pogobatch generates "
            "Rundra application preparation. For versioned pogosim-full_vX.Y.Z.sif "
            "images, the official Library URI is inferred automatically."
        ),
    )
    cluster.add_argument(
        "--container-sha256",
        default=None,
        help=(
            "Expected SHA-256 for --container-uri. If --container-image names an "
            "existing local file, Pogobatch computes this automatically."
        ),
    )
    cluster.add_argument(
        "--prepare-simulator",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "For remote generated campaigns, compile a simulator with an adjacent "
            "Makefile once in Rundra preparation (default: enabled). Use "
            "--no-prepare-simulator only when the executable is deliberately staged "
            "or already present in the runtime image."
        ),
    )
    cluster.add_argument(
        "--auto-apptainer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "On CONTAINER_REQUIRED, automatically create a temporary Rundra v5 "
            "preparation recipe from pogosim-apptainer.def (default: enabled). "
            "Use --no-auto-apptainer to require an explicit image/project recipe."
        ),
    )
    cluster.add_argument(
        "--apptainer-def",
        type=Path,
        default=None,
        help=(
            "Pogosim Apptainer definition used by automatic remote preparation. "
            "Relative paths are resolved from --source-root; default: "
            "pogosim-apptainer.def."
        ),
    )
    cluster.add_argument("--image-build-cpus", type=int, default=4)
    cluster.add_argument("--image-build-memory", default="8GiB")
    cluster.add_argument("--image-build-walltime", default="01:00:00")
    cluster.add_argument(
        "--sim-build-cpus",
        type=int,
        default=1,
        help=(
            "CPUs for the Rundra preparation allocation used to compile the "
            "selected simulator with make ... sim (default: 1)."
        ),
    )
    cluster.add_argument(
        "--sim-build-memory",
        default="2GiB",
        help="Memory for simulator application preparation (default: 2GiB).",
    )
    cluster.add_argument(
        "--sim-build-walltime",
        default="00:15:00",
        help=(
            "Walltime for simulator application preparation (default: 00:15:00)."
        ),
    )
    cluster.add_argument("--container-gpu", action="store_true")
    cluster.add_argument("--json", action="store_true")
    cluster.set_defaults(handler=command_cluster)

    return parser


def _configure_logging(verbose: bool, quiet: bool) -> None:
    # Keep Pogosim's normal logging setup, then narrow Pogobatch's own level.
    utils.init_logging(verbose)
    pogobatch_logger = logging.getLogger("pogobatch")
    pogobatch_logger.handlers.clear()
    pogobatch_logger.propagate = True
    if verbose:
        pogobatch_logger.setLevel(logging.DEBUG)
    elif quiet:
        pogobatch_logger.setLevel(logging.ERROR)
    else:
        pogobatch_logger.setLevel(logging.WARNING)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(
        getattr(args, "verbose", False),
        getattr(args, "quiet", False),
    )

    try:
        return int(args.handler(args))
    except PogobatchError as exc:
        logger.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        logger.error("Interrupted")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())

# MODELINE "{{{1
# vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
# vim:foldmethod=marker
