from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
import textwrap
import unittest
from unittest import mock

from pogosim import pogobatch


class LocalCampaignApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.config = self.root / "config.yaml"
        self.config.write_text("seed: 0\n", encoding="utf-8")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_success_returns_outputs_and_manifest(self) -> None:
        output_dir = self.root / "out"

        def fake_merge(_artifacts, destination, **_kwargs):
            destination.mkdir(parents=True, exist_ok=True)
            output = destination / "result.feather"
            output.touch()
            return [output]

        successful = [{"ok": True, "manifest": "unused"} for _ in range(2)]
        with (
            mock.patch.object(pogobatch, "_execute_local_specs", return_value=successful),
            mock.patch.object(pogobatch, "discover_task_artifacts", return_value=[]),
            mock.patch.object(pogobatch, "merge_task_artifacts", side_effect=fake_merge),
        ):
            result = pogobatch.run_local_campaign(
                self.config,
                "simulator",
                seeds=(10, 11),
                output_dir=output_dir,
                temp_base=self.root / "tmp",
                backend="sequential",
            )

        self.assertEqual(result.task_count, 2)
        self.assertEqual(result.seeds, (10, 11))
        self.assertIsNone(result.retained_temp_dir)
        self.assertEqual(result.outputs, (output_dir / "result.feather",))
        manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["seeds"], [10, 11])
        self.assertEqual(manifest["backend"], "sequential")

    def test_task_failure_is_structured_and_retains_campaign(self) -> None:
        failure = {
            "ok": False,
            "combination_ordinal": 0,
            "logical_run": 0,
            "seed": 10,
            "error": "crashed",
        }
        with mock.patch.object(
            pogobatch, "_execute_local_specs", return_value=[failure]
        ):
            with self.assertRaises(pogobatch.LocalCampaignError) as caught:
                pogobatch.run_local_campaign(
                    self.config,
                    "simulator",
                    seeds=(10,),
                    output_dir=self.root / "out",
                    temp_base=self.root / "tmp",
                    backend="sequential",
                )

        self.assertTrue(caught.exception.campaign_dir.is_dir())
        self.assertEqual(caught.exception.failures[0]["error"], "crashed")

    def test_rejects_duplicate_or_out_of_range_seeds(self) -> None:
        common = dict(
            config_path=self.config,
            simulator_binary="simulator",
            output_dir=self.root / "out",
            temp_base=self.root / "tmp",
        )
        with self.assertRaises(pogobatch.PogobatchError):
            pogobatch.run_local_campaign(seeds=(1, 1), **common)
        with self.assertRaises(pogobatch.PogobatchError):
            pogobatch.run_local_campaign(seeds=(2**32,), **common)

    def test_runtime_config_isolates_flash_output_per_task(self) -> None:
        source = {
            "data_filename": "frames/data.feather",
            "flash_state": {
                "input_file": "starting.pgflash",
                "output_file": "checkpoints/final.pgflash",
            },
        }
        runtime, _ = pogobatch._prepare_runtime_config(
            source, self.root / "task"
        )

        self.assertEqual(
            runtime["flash_state"]["input_file"], "starting.pgflash"
        )
        self.assertEqual(
            runtime["flash_state"]["output_file"],
            str(self.root / "task/frames/final.pgflash"),
        )
        self.assertEqual(
            source["flash_state"]["output_file"],
            "checkpoints/final.pgflash",
        )

    def test_final_failed_attempt_is_retained_for_diagnosis(self) -> None:
        spec = pogobatch.LocalTaskSpec(
            combination_ordinal=0,
            config={"seed": 0},
            result_filename="result.feather",
            extra_columns={},
            config_hash="hash",
            logical_run=0,
            base_seed=5,
            all_seeds=(5,),
            simulator_binary="simulator",
            campaign_dir=str(self.root / "campaign"),
            gui=False,
            max_retries=1,
            retry_new_seed=True,
            simulator_output="quiet",
        )
        with mock.patch.object(
            pogobatch,
            "execute_task",
            side_effect=subprocess.CalledProcessError(2, ["simulator"]),
        ):
            result = pogobatch._run_local_task_worker(spec)
        final_attempt = self.root / "campaign/combo_000000/run_000000_seed_6_try_1"
        self.assertFalse(result["ok"])
        self.assertTrue((final_attempt / "input_config.yaml").is_file())

    def test_end_to_end_with_fake_simulator(self) -> None:
        simulator = self.root / "fake_simulator.py"
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
                output = Path(config["data_filename"])
                output.parent.mkdir(parents=True, exist_ok=True)
                feather.write_feather(
                    pa.table({"time": [0.0], "robot_category": ["robots"],
                              "robot_id": [0], "x": [0.0], "y": [0.0],
                              "angle": [0.0]}),
                    output,
                )
                """
            ),
            encoding="utf-8",
        )
        simulator.chmod(0o755)
        self.config.write_text(
            textwrap.dedent(
                """\
                data_filename: frames/data.feather
                choice:
                  default_option: 1
                  batch_options: [1, 2]
                pogobatch:
                  result_filename_format: result_{choice}.feather
                  result_new_columns: [choice]
                """
            ),
            encoding="utf-8",
        )
        result = pogobatch.run_local_campaign(
            self.config,
            str(simulator),
            seeds=(3, 4),
            output_dir=self.root / "out",
            temp_base=self.root / "tmp",
            backend="sequential",
        )
        self.assertEqual(result.task_count, 4)
        self.assertEqual(len(result.outputs), 2)
        for output in result.outputs:
            import pandas as pd

            frame = pd.read_feather(output)
            self.assertEqual(set(frame["seed"]), {3, 4})
            self.assertIn("choice", frame.columns)


if __name__ == "__main__":
    unittest.main()
