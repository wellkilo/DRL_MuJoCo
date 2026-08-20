from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from drl.config_loader import load_config
from drl.training_utils import average_metrics, resolve_training_topology


class TrainingTopologyTest(unittest.TestCase):
    def test_single_config_really_uses_one_actor(self) -> None:
        config = load_config("config/config_single.yaml")
        topology = resolve_training_topology(
            num_actors=config.num_actors,
            num_gpus=config.num_gpus,
            actors_per_gpu=config.actors_per_gpu,
            available_gpus=1,
        )
        self.assertEqual(topology.total_actors, 1)
        self.assertEqual(topology.actors_per_gpu, 1)

    def test_scaling_config_preserves_explicit_topology(self) -> None:
        config = load_config("config/scaling/hopper_gpu4.yaml")
        topology = resolve_training_topology(
            num_actors=config.num_actors,
            num_gpus=config.num_gpus,
            actors_per_gpu=config.actors_per_gpu,
            available_gpus=4,
        )
        self.assertEqual(topology.active_gpus, 4)
        self.assertEqual(topology.total_actors, 32)

    def test_all_scaling_configs_have_consistent_topology(self) -> None:
        for path in sorted(Path("config/scaling").glob("*.yaml")):
            with self.subTest(config=str(path)):
                config = load_config(str(path))
                topology = resolve_training_topology(
                    num_actors=config.num_actors,
                    num_gpus=config.num_gpus,
                    actors_per_gpu=config.actors_per_gpu,
                    available_gpus=config.num_gpus,
                )
                self.assertEqual(topology.total_actors, config.num_actors)

    def test_mismatched_explicit_topology_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not match"):
            resolve_training_topology(
                num_actors=7,
                num_gpus=2,
                actors_per_gpu=4,
                available_gpus=2,
            )

    def test_cpu_fallback_keeps_actors_per_learner_bounded(self) -> None:
        topology = resolve_training_topology(
            num_actors=32,
            num_gpus=4,
            actors_per_gpu=8,
            available_gpus=0,
        )
        self.assertEqual(topology.active_gpus, 1)
        self.assertEqual(topology.total_actors, 8)


class ConfigLoadingTest(unittest.TestCase):
    def test_missing_config_is_rejected(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_config("config/does-not-exist.yaml")

    def test_non_mapping_config_is_rejected(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "invalid.yaml"
            path.write_text("- not\n- a\n- mapping\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "YAML mapping"):
                load_config(str(path))


class MetricsAggregationTest(unittest.TestCase):
    def test_metrics_are_averaged_across_available_learners(self) -> None:
        metrics = average_metrics(
            [{"loss": 1.0, "lr": 0.1}, {"loss": 3.0}, {"loss": 5.0, "lr": 0.3}]
        )
        self.assertEqual(metrics, {"loss": 3.0, "lr": 0.2})


if __name__ == "__main__":
    unittest.main()
