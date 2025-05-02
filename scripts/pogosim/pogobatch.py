#!/usr/bin/env python3
import os
import sys
import subprocess
import yaml
import argparse
import itertools
import tempfile
import shutil
import logging
import pandas as pd
import copy
import re
import string
from typing import Any, Dict, Iterable, List
import uuid

from pogosim import utils
from pogosim import __version__

# Import Pool from multiprocessing for the default backend.
from multiprocessing import Pool


def set_in_dict(d: Dict[str, Any], dotted_key: str, value: Any, sep: str = ".") -> None:
    """
    Like d['a']['b']['c'] = value but with a single dotted string.
    Creates intermediate dictionaries if needed.
    """
    keys = dotted_key.split(sep)
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


class DotDict(dict):
    """Allow attribute access (`obj.key`) as an alias for mapping access (`obj['key']`)."""

    # Read
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    # Write
    def __setattr__(self, key, value):
        self[key] = value


def to_dotdict(obj):
    """Recursively convert every plain dict in *obj* into a DotDict."""
    if isinstance(obj, dict):
        return DotDict({k: to_dotdict(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [to_dotdict(v) for v in obj]
    return obj


def get_by_dotted_path(node: dict, dotted: str, sep: str = "."):
    """Return cfg['a']['b']['c'] when dotted == 'a.b.c'."""
    for part in dotted.split(sep):
        node = node[part]
    return node


class PogobotLauncher:
    def __init__(self, num_instances, base_config_path, combined_output_path, simulator_binary, temp_base_path, backend="multiprocessing", keep_temp=False, extra_columns=None, max_retries: int=5):
        self.num_instances = num_instances
        self.base_config_path = base_config_path
        self.combined_output_path = combined_output_path
        self.simulator_binary = simulator_binary
        self.temp_base_path = temp_base_path
        self.backend = backend
        self.keep_temp = keep_temp
        self.temp_dirs = []
        self.dataframes = []  # Will hold DataFrames loaded from each run
        self.extra_columns = extra_columns or {}
        self.max_retries = max_retries

    @staticmethod
    def modify_config_static(base_config_path, output_dir, seed):
        # Load the base YAML configuration.
        with open(base_config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Set a unique seed for this instance.
        config['seed'] = seed

        ## Disable frame export.
        #config['save_video_period'] = -1

        # Create a directory for frame files inside the temporary directory.
        frames_dir = os.path.join(output_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)

        # Update file paths so that outputs go into the temporary directory.
        if 'data_filename' in config:
            config['data_filename'] = os.path.join(frames_dir, os.path.basename(config['data_filename']))
        if 'console_filename' in config:
            config['console_filename'] = os.path.join(frames_dir, os.path.basename(config['console_filename']))
        if 'frames_name' in config:
            config['frames_name'] = os.path.join(frames_dir, os.path.basename(config['frames_name']))

        # Write the modified configuration to a new YAML file.
        new_config_path = os.path.join(output_dir, "test.yaml")
        with open(new_config_path, 'w') as f:
            yaml.safe_dump(config, f)

        return new_config_path

    @staticmethod
    def launch_simulator_static(config_path, simulator_binary):
        # Build the simulator command and run it.
        command = [simulator_binary, "-c", config_path, "-nr", "-g", "-q"]
        logging.debug(f"Executing command: {' '.join(command)}")
        subprocess.run(command, check=True)

    @staticmethod
    def worker(args):
        (i, base_cfg, sim_bin, tmp_base, max_retries) = args
        attempt = 0
        while True:
            # create a fresh sub-directory for every attempt
            temp_dir = tempfile.mkdtemp(
                prefix=f"sim_{i}_try{attempt}_", dir=tmp_base)

            try:
                cfg_path = PogobotLauncher.modify_config_static(
                    base_cfg, temp_dir, seed=i)
                PogobotLauncher.launch_simulator_static(cfg_path, sim_bin)
                break                       # success
            except subprocess.CalledProcessError as exc:
                logging.warning(
                    "Instance %d – crash on attempt %d/%d: %s",
                    i, attempt + 1, max_retries, exc)
                shutil.rmtree(temp_dir, ignore_errors=True)
                attempt += 1
                if attempt > max_retries:
                    logging.error("Instance %d – gave up after %d retries",
                                  i, max_retries)
                    return (None, None)     # hard failure
                continue                    # retry

        # Load the Feather file as soon as the simulator instance finishes.
        feather_path = os.path.join(temp_dir, "frames", "data.feather")
        df = None
        if os.path.exists(feather_path):
            try:
                df = pd.read_feather(feather_path)
                # Add a column "run" corresponding to the instance number.
                df['run'] = i
                logging.debug(f"Instance {i}: Loaded data from {feather_path}")
            except Exception as e:
                logging.error(f"Instance {i}: Error reading feather file {feather_path}: {e}")
        else:
            logging.warning(f"Instance {i}: Feather file not found: {feather_path}")
        return (temp_dir, df)

    def combine_feather_files(self, dataframes):
        if dataframes:
            combined_df = pd.concat(dataframes, ignore_index=True)
            combined_df.to_feather(self.combined_output_path)
            logging.info(f"Combined data saved to {self.combined_output_path}")
        else:
            logging.error("No dataframes were loaded to combine.")

    def clean_temp_dirs(self):
        for d in self.temp_dirs:
            shutil.rmtree(d)
            logging.debug(f"Cleaned up temporary directory: {d}")

    def launch_all(self):
        # Prepare the arguments for each simulation instance.
        args_list = [
            (i, self.base_config_path, self.simulator_binary, self.temp_base_path, self.max_retries)
            for i in range(self.num_instances)
        ]

        if self.backend == "multiprocessing":
            # Use a multiprocessing Pool.
            with Pool(processes=self.num_instances) as pool:
                results = pool.map(PogobotLauncher.worker, args_list)
        elif self.backend == "ray":
            try:
                import ray
            except ImportError:
                logging.error("Ray is not installed. Please install ray to use the 'ray' backend.")
                sys.exit(1)
            # Initialize ray.
            ray.init(ignore_reinit_error=True)
            # Convert the worker function into a Ray remote function.
            ray_worker = ray.remote(PogobotLauncher.worker)
            futures = [ray_worker.remote(args) for args in args_list]
            results = ray.get(futures)
            ray.shutdown()
        else:
            logging.error(f"Unknown backend: {self.backend}")
            sys.exit(1)

        # Separate the temporary directories and the loaded DataFrames.
        self.temp_dirs = [result[0] for result in results]
        self.dataframes = [result[1] for result in results if result[1] is not None]

        # Inject the EXTRA columns (same constant value for every row)
        if self.extra_columns and self.dataframes:
            for df in self.dataframes:
                for col, val in self.extra_columns.items():
                    df[col] = val

        # Combine the loaded DataFrames.
        self.combine_feather_files(self.dataframes)

        if not self.keep_temp:
            self.clean_temp_dirs()
        else:
            logging.info("Keeping temporary directories:")
            for d in self.temp_dirs:
                logging.info(d)

class PogobotBatchRunner:
    """
    A reusable class to run batch simulations for every combination of parameters
    specified in a multi-value YAML configuration file. The class computes all
    combinations, writes a temporary YAML file for each combination (in a specified
    temporary base directory), computes a friendly output filename, and launches
    a PogobotLauncher for each combination.
    """
    def __init__(self, multi_config_file, runs, simulator_binary, temp_base, output_dir,
                 backend="multiprocessing", keep_temp=False, verbose=False, retries: int = 5):
        self.multi_config_file = multi_config_file
        self.runs = runs
        self.simulator_binary = simulator_binary
        self.temp_base = temp_base
        self.output_dir = output_dir
        self.backend = backend
        self.keep_temp = keep_temp
        self.verbose = verbose
        self.retries = retries

        # Initialize logging via utils.
        utils.init_logging(self.verbose)


    def get_combinations(self, config: dict) -> list[dict]:
        """
        Return every configuration obtained by choosing **exactly one** value
        from each *batch_options* list found anywhere in the YAML tree.

        • Ordinary Python lists that are *not* under a `batch_options` key are
          treated as plain data (e.g. vectors, colour tables) and **do not**
          create extra combinations.

        • If no batch_options are present, the original config is returned.
        """
        option_paths: list[str] = []
        option_values: list[list] = []

        # ---------------------------------------------------------------------
        # Collect every "...batch_options: [v1, v2, ...]" we can find
        # ---------------------------------------------------------------------
        def recurse(node: dict | list | Any, prefix: str = "") -> None:
            if isinstance(node, dict):
                if "batch_options" in node and isinstance(node["batch_options"], list):
                    option_paths.append(prefix.rstrip("."))       # where to set
                    option_values.append(node["batch_options"])   # the choices
                    return                                        # don't dive deeper
                for k, v in node.items():
                    recurse(v, f"{prefix}{k}.")
            # NOTE: plain lists are *ignored* on purpose.

        recurse(config)

        # ---------------------------------------------------------------------
        # Build the Cartesian product of every collected factor
        # ---------------------------------------------------------------------
        if not option_paths:                       # nothing to vary
            return [config]

        combos: list[dict] = []
        for prod in itertools.product(*option_values):
            cfg = copy.deepcopy(config)
            for path, val in zip(option_paths, prod, strict=True):
                set_in_dict(cfg, path, val)        # overwrite dict→raw value
            combos.append(cfg)
        return combos


    def write_temp_yaml(self, comb_config):
        """
        Write the combination configuration dictionary to a temporary YAML file in temp_base.
        Returns the path to the file.
        """
        tmp_file = tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".yaml", prefix="combo_", encoding="utf-8", dir=self.temp_base
        )
        yaml.safe_dump(comb_config, tmp_file)
        tmp_file.close()
        logging.debug(f"Wrote temporary YAML config: {tmp_file.name}")
        return tmp_file.name


    def compute_output_filename(self, comb_config: dict) -> str:
        fmt = comb_config.get("result_filename_format")
        if not fmt:
            # exactly the same default as before
            return (os.path.join(self.output_dir, "result.feather")
                    if self.output_dir else "result.feather")

        # ── 1. deep‑copy and basename‑shorten all path‑like strings ──────────
        def normalise(node):
            if isinstance(node, dict):
                return {k: normalise(v) for k, v in node.items()}
            if isinstance(node, list):
                return [normalise(v) for v in node]
            if isinstance(node, str) and os.path.dirname(node):
                return os.path.splitext(os.path.basename(node))[0]
            return node

        cfg_for_fmt = normalise(copy.deepcopy(comb_config))

        # ── 2. wrap every dict so dots act like nested keys ──────────────────
        dot_cfg = to_dotdict(cfg_for_fmt)

        # ── 3. run *standard* str.format_map (no custom formatter needed) ────
        try:
            filename = fmt.format_map(dot_cfg)
        except KeyError as exc:
            logging.error("Error formatting result filename: missing key %s", exc)
            filename = "result.feather"
        except Exception as exc:                    # noqa: BLE001
            logging.error("Error formatting result filename: %s", exc)
            filename = "result.feather"

        # ── 4. anchor in output directory if relative ────────────────────────
        if self.output_dir and not os.path.isabs(filename):
            filename = os.path.join(self.output_dir, filename)
        return filename


    def run_launcher_for_combination(self,
                                     temp_config_path: str,
                                     final_output: str,
                                     comb_config: dict) -> str:
        # ── Build extra-columns dict (same as before) ──────────────────────────
        extra_columns: dict[str, Any] = {}
        for path in comb_config.get("result_new_columns", []):
            try:
                raw = get_by_dotted_path(comb_config, path)
                if isinstance(raw, str) and os.path.dirname(raw):
                    raw = os.path.splitext(os.path.basename(raw))[0]
                extra_columns[path] = raw
            except KeyError:
                logging.error("result_new_columns: path '%s' not found", path)

        # ── 1. Choose a *temporary* output Feather for this run ────────────────
        tmp_output = os.path.join(self.temp_base,
                                  f"run_{uuid.uuid4().hex}.feather")

        logging.info("Launch → tmp %s  (will merge into %s)",
                     tmp_output, final_output)

        # ── 2. Run the simulator ------------------------------------------------
        launcher = PogobotLauncher(
            num_instances        = self.runs,
            base_config_path     = temp_config_path,
            combined_output_path = tmp_output,
            simulator_binary     = self.simulator_binary,
            temp_base_path       = self.temp_base,
            backend              = self.backend,
            keep_temp            = self.keep_temp,
            extra_columns        = extra_columns,
            max_retries          = self.retries
        )
        launcher.launch_all()
        os.remove(temp_config_path)

        # ── 3. Merge tmp → final (append or create) ----------------------------
        try:
            new_df = pd.read_feather(tmp_output)
            if os.path.exists(final_output):
                old_df = pd.read_feather(final_output)
                pd.concat([old_df, new_df], ignore_index=True).to_feather(final_output)
                logging.info("Appended %d rows to %s", len(new_df), final_output)
            else:
                new_df.to_feather(final_output)
                logging.info("Created %s with %d rows", final_output, len(new_df))
        finally:
            os.remove(tmp_output)                      # always clean up temp

        return final_output

    def run_all(self):
        """
        Load the multi-value YAML configuration, compute combinations, write temporary files,
        launch a PogobotLauncher for each combination sequentially, and return a list of output files.
        """
        # Ensure the temporary base and output directories exist.
        if not os.path.exists(self.temp_base):
            os.makedirs(self.temp_base, exist_ok=True)
            logging.info(f"Created temporary base directory: {self.temp_base}")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
            logging.info(f"Created output directory: {self.output_dir}")

        # Load the multi-value configuration.
        with open(self.multi_config_file, "r") as f:
            multi_config = yaml.safe_load(f)

        combinations = self.get_combinations(multi_config)
        if not combinations:
            logging.error("No combinations found in the configuration.")
            sys.exit(1)
        logging.info(f"Found {len(combinations)} combination(s) to run.")

        tasks = []
        for comb in combinations:
            temp_yaml = self.write_temp_yaml(comb)
            output_file = self.compute_output_filename(comb)
            tasks.append((temp_yaml, output_file, comb))
            logging.info(f"Task: Config file {temp_yaml} -> Output: {output_file}")

        # Remove any pre-existing result_*.feather before first append
        for outfile in {t[1] for t in tasks}:                   # unique names
            if os.path.exists(outfile):
                os.remove(outfile)
                logging.info("Removed stale result file: %s", outfile)

        results = []
        for temp_yaml, output_file, comb in tasks:
            result = self.run_launcher_for_combination(temp_yaml, output_file, comb)
            results.append(result)

        logging.info("Batch run completed. Generated output files:")
        for output_file in results:
            logging.info(f" - {output_file}")
        return results

def main():
    parser = argparse.ArgumentParser(
        description="Batch run multiple PogobotLauncher instances sequentially for every combination of parameters specified in a multi-value YAML config."
    )
    parser.add_argument("-c", "--config", type=str, default="",
                        help="Path to the YAML configuration file with multiple choices (lists) for some parameters.")
    parser.add_argument("-r", "--runs", type=int, default=1,
                        help="Number of simulator runs to launch per configuration combination (default: 1).")
    parser.add_argument("-S", "--simulator-binary", type=str, default="",
                        help="Path to the simulator binary.")
    parser.add_argument("-t", "--temp-base", type=str, default="tmp",
                        help="Base directory for temporary directories and YAML config files used by PogobotLauncher (default: 'tmp').")
    parser.add_argument("-o", "--output-dir", type=str, default=".",
                        help="Directory where the combined output Feather files will be saved (default: current directory).")
    parser.add_argument("--backend", choices=["multiprocessing", "ray"], default="multiprocessing",
                        help="Parallelism backend to use for launching PogobotLauncher instances (default: multiprocessing).")
    parser.add_argument("--keep-temp", action="store_true",
                        help="Keep temporary directories after simulation runs.")
    parser.add_argument("-v", "--verbose", default=False, action="store_true", help="Verbose mode")
    parser.add_argument("-V", "--version", default=False, action="store_true", help="Return version")
    parser.add_argument("-R", "--retries", type=int, default=5, help="How many times to relaunch a run when the simulator crashes (default: 5).")
    args = parser.parse_args()

    if args.version:
        print(f"Pogosim version {__version__}")
        sys.exit(0)

    if not len(args.config) or not len(args.simulator_binary):
        parser.print_usage()
        sys.exit(1)

    runner = PogobotBatchRunner(
        multi_config_file=args.config,
        runs=args.runs,
        simulator_binary=args.simulator_binary,
        temp_base=args.temp_base,
        output_dir=args.output_dir,
        backend=args.backend,
        keep_temp=args.keep_temp,
        verbose=args.verbose,
        retries=args.retries
    )
    runner.run_all()

if __name__ == "__main__":
    main()

# MODELINE "{{{1
# vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
# vim:foldmethod=marker
