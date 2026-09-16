from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import tempfile
import textwrap
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from pogosim import pogobatch, pogoptim


class PogoptimConfigurationTests(unittest.TestCase):
    def test_yaml_defaults_and_explicit_cli_overrides(self) -> None:
        config = {
            "optimization": {
                "algorithm": "mapelites",
                "budget": 9,
                "parallelism": {"candidate_jobs": 3, "batch_jobs": 2},
                "qd": {
                    "shape": [4, 5],
                    "features_domain": [[-1, 1], [0, 10]],
                },
            }
        }
        cli = argparse.Namespace(optimizer="random", max_evals=4)
        settings = pogoptim.resolve_optimization_settings(config, cli)
        self.assertEqual(settings.algorithm, "random")
        self.assertEqual(settings.budget, 4)
        self.assertEqual(settings.candidate_jobs, 3)
        self.assertEqual(settings.qd_shape, (4, 5))
        self.assertEqual(settings.qd_features_domain, ((-1.0, 1.0), (0.0, 10.0)))

    def test_shape_and_feature_dimensions_must_match(self) -> None:
        config = {
            "optimization": {
                "qd": {
                    "shape": [4],
                    "features_domain": [[0, 1], [0, 1]],
                }
            }
        }
        with self.assertRaisesRegex(RuntimeError, "same dimension"):
            pogoptim.resolve_optimization_settings(config, argparse.Namespace())

    def test_candidate_seed_blocks_are_deterministic_and_disjoint(self) -> None:
        kwargs = dict(
            optimizer_seed=7,
            evaluations=4,
            runs=3,
            retries=2,
            retry_new_seed=True,
        )
        first = pogoptim.CandidateSeedAllocator(**kwargs)
        second = pogoptim.CandidateSeedAllocator(**kwargs)
        blocks = [first.seeds_for(index) for index in range(4)]
        self.assertEqual(blocks, [second.seeds_for(index) for index in range(4)])
        self.assertEqual(len(set().union(*map(set, blocks))), 12)
        # Pogobatch retry seeds occupy the reserved remainder of each block.
        for index, seeds in enumerate(blocks):
            retry_seed = pogobatch._replacement_seed(seeds[0], 0, seeds, 1)
            next_start = first.offset + (index + 1) * first.block_size
            self.assertLess(retry_seed, next_start)


class PogoptimEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.specs = [pogoptim.VarSpec("gain", "float", 0.0, 1.0)]
        self.arguments = dict(
            index=0,
            u_internal=np.array([0.5]),
            specs=self.specs,
            base_cfg={"gain": 0.5},
            simulator_binary="simulator",
            seeds=(10,),
            temp_base="tmp",
            backend="sequential",
            batch_jobs=1,
            keep_temp=False,
            retries=0,
            retry_new_seed=True,
            default_features_fn=lambda _df: np.array([0.25, 0.75]),
            qd_mode=True,
            custom_objective=False,
            qd_domains_explicit=False,
            feature_domains=((0.0, 1.0), (0.0, 1.0)),
        )

    def test_expected_campaign_failure_is_penalized(self) -> None:
        failure = pogobatch.LocalCampaignError(
            "crashed", campaign_dir=Path("tmp/failure"), failures=({"error": "crashed"},)
        )
        with mock.patch.object(pogoptim, "run_evaluation", side_effect=failure):
            result = pogoptim.evaluate_candidate(
                objective_fn=lambda _df: 1.0, **self.arguments
            )
        self.assertFalse(result.success)
        self.assertEqual(result.fitness, pogoptim.PENALTY)

    def test_custom_descriptors_require_explicit_domains(self) -> None:
        frame = pd.DataFrame({"seed": [10], "run": [0]})
        arguments = dict(self.arguments)
        arguments["custom_objective"] = True
        with mock.patch.object(pogoptim, "run_evaluation", return_value=frame):
            with self.assertRaises(pogoptim.ObjectiveContractError):
                pogoptim.evaluate_candidate(
                    objective_fn=lambda _df: (1.0, [0.2, 0.8]), **arguments
                )

    def test_out_of_domain_descriptors_fail_without_clipping(self) -> None:
        frame = pd.DataFrame({"seed": [10], "run": [0]})
        arguments = dict(self.arguments)
        arguments.update(custom_objective=True, qd_domains_explicit=True)
        with mock.patch.object(pogoptim, "run_evaluation", return_value=frame):
            result = pogoptim.evaluate_candidate(
                objective_fn=lambda _df: (1.0, [2.0, 0.8]), **arguments
            )
        self.assertFalse(result.success)
        self.assertIn("outside", result.error or "")


class PogoptimEndToEndTests(unittest.TestCase):
    def test_random_search_uses_current_pogobatch_api(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            simulator = root / "fake_simulator.py"
            simulator.write_text(
                textwrap.dedent(
                    """\
                    #!/usr/bin/env python3
                    import argparse
                    from pathlib import Path
                    import pyarrow as pa
                    import pyarrow.feather as feather
                    import yaml

                    parser = argparse.ArgumentParser(add_help=False)
                    parser.add_argument("-c")
                    parser.add_argument("--seed", type=int)
                    args, _ = parser.parse_known_args()
                    config = yaml.safe_load(Path(args.c).read_text(encoding="utf-8"))
                    if "optimization" in config or "pogobatch" in config:
                        raise SystemExit(3)
                    output = Path(config["data_filename"])
                    output.parent.mkdir(parents=True, exist_ok=True)
                    feather.write_feather(pa.table({
                        "time": [0.0, 1.0], "robot_category": ["robots", "robots"],
                        "robot_id": [0, 0], "x": [0.0, 1.0], "y": [0.0, 0.0],
                        "angle": [0.0, 0.0],
                    }), output)
                    """
                ),
                encoding="utf-8",
            )
            simulator.chmod(0o755)
            objective = root / "objective.py"
            objective.write_text(
                "def compute_objective(df):\n    return float(len(df))\n",
                encoding="utf-8",
            )
            output_dir = root / "opt_out"
            temp_dir = root / "tmp"
            config = root / "optim.yaml"
            config.write_text(
                textwrap.dedent(
                    f"""\
                    optimization:
                      algorithm: random
                      seed: 7
                      budget: 2
                      runs: 1
                      retries: 0
                      backend: sequential
                      parallelism:
                        candidate_jobs: 2
                        batch_jobs: 1
                      output_dir: {output_dir}
                      temp_base: {temp_dir}
                      objective:
                        path: {objective}
                    data_filename: frames/data.feather
                    gain:
                      default_option: 0.5
                      optimization_domain: {{type: float, min: 0.0, max: 1.0}}
                    """
                ),
                encoding="utf-8",
            )

            status = pogoptim.main([
                "-c", str(config), "-S", str(simulator),
                "--backend", "multiprocessing",
            ])
            self.assertEqual(status, 0)
            history = pd.read_csv(output_dir / "opt_history.csv")
            self.assertEqual(list(history["status"]), ["success", "success"])
            self.assertNotEqual(history.loc[0, "base_seeds"], history.loc[1, "base_seeds"])
            self.assertTrue((output_dir / "best_results.feather").is_file())
            self.assertTrue((output_dir / "summary.json").is_file())

    def test_all_failures_write_summary_and_worker_counts_are_capped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "optim.yaml"
            config.write_text(
                "gain:\n  default_option: 0.5\n"
                "  optimization_domain: {type: float, min: 0, max: 1}\n",
                encoding="utf-8",
            )
            settings = pogoptim.OptimizationSettings(
                algorithm="random",
                budget=2,
                candidate_jobs=4,
                batch_jobs=4,
                output_dir=str(root / "out"),
                temp_base=str(root / "tmp"),
            )

            def failed_candidate(**kwargs):
                return pogoptim.EvaluationResult(
                    index=kwargs["index"],
                    u_internal=kwargs["u_internal"],
                    values={"gain": 0.5},
                    base_seeds=kwargs["seeds"],
                    effective_seeds=kwargs["seeds"],
                    error="expected failure",
                )

            with (
                mock.patch.object(pogoptim, "_available_cpu_count", return_value=2),
                mock.patch.object(pogoptim, "evaluate_candidate", side_effect=failed_candidate),
            ):
                with self.assertRaisesRegex(RuntimeError, "No successful evaluations"):
                    pogoptim.optimize(str(config), "/bin/true", settings)

            import json

            summary = json.loads((root / "out" / "summary.json").read_text())
            self.assertEqual(summary["status"], "failed")
            self.assertEqual(summary["parallelism"]["candidate_jobs"], 2)
            self.assertEqual(summary["parallelism"]["batch_jobs"], 1)


@unittest.skipUnless(importlib.util.find_spec("cma"), "CMA is optional")
class CmaEsSmokeTests(unittest.TestCase):
    def test_cmaes_uses_batched_evaluations(self) -> None:
        optimizer = pogoptim.CMAES(
            dim=2,
            u0=np.array([0.5, 0.5]),
            sigma0=0.2,
            popsize=2,
            seed=7,
            max_evals=4,
        )
        batch_sizes = []

        def evaluate_many(points):
            batch_sizes.append(len(points))
            return [-float(np.square(point - 0.25).sum()) for point in points]

        optimizer.run(evaluate_many)
        self.assertEqual(sum(batch_sizes), 4)
        self.assertIsNotNone(optimizer.best_u)


@unittest.skipUnless(importlib.util.find_spec("qdpy"), "QDpy is optional")
class MapElitesSmokeTests(unittest.TestCase):
    def test_map_elites_writes_archive_and_pickle(self) -> None:
        specs = [
            pogoptim.VarSpec("x", "float", 0.0, 1.0),
            pogoptim.VarSpec("y", "float", 0.0, 1.0),
        ]
        next_index = 0

        def evaluate_many(points):
            nonlocal next_index
            results = []
            for point in points:
                results.append(
                    pogoptim.EvaluationResult(
                        index=next_index,
                        u_internal=np.asarray(point),
                        values=pogoptim.decode_unit_vector(specs, np.asarray(point)),
                        base_seeds=(1,),
                        effective_seeds=(1,),
                        fitness=float(sum(point)),
                        features=np.asarray(point),
                        success=True,
                    )
                )
                next_index += 1
            return results

        with tempfile.TemporaryDirectory() as directory:
            with mock.patch("qdpy.plots.default_plots_grid"):
                info = pogoptim.run_qdpy_map_elites(
                    specs,
                    evaluate_many,
                    (2, 2),
                    ((0.0, 1.0), (0.0, 1.0)),
                    2,
                    4,
                    42,
                    directory,
                )
            self.assertGreater(info["elite_count"], 0)
            self.assertTrue((Path(directory) / "qd_archive.csv").is_file())
            self.assertTrue((Path(directory) / "qd_final.p").is_file())


if __name__ == "__main__":
    unittest.main()
