from __future__ import annotations

import unittest

import numpy as np
import torch

from drl.models import ActorCritic
from drl.rollout_utils import (
    compute_batch_moments,
    compute_gae,
    merge_rms_states,
    prepare_action,
)


class ActionContractTest(unittest.TestCase):
    def test_raw_policy_action_keeps_old_log_probability_consistent(self) -> None:
        torch.manual_seed(42)
        model = ActorCritic(3, 2)
        observations = torch.zeros((2_000, 3))
        distribution, _ = model.get_dist_and_value(observations)
        sampled = distribution.sample()
        old_log_probability = distribution.log_prob(sampled).sum(axis=-1)

        policy_action, env_action = prepare_action(
            sampled.numpy(),
            np.full(2, -0.25, dtype=np.float32),
            np.full(2, 0.25, dtype=np.float32),
        )
        recomputed = distribution.log_prob(torch.from_numpy(policy_action)).sum(axis=-1)

        self.assertTrue(np.any(policy_action != env_action))
        torch.testing.assert_close(recomputed, old_log_probability)
        self.assertTrue(np.all(env_action >= -0.25))
        self.assertTrue(np.all(env_action <= 0.25))


class GaeContractTest(unittest.TestCase):
    def test_termination_does_not_bootstrap(self) -> None:
        advantages, returns = compute_gae(
            [1.0],
            [2.0],
            [True],
            [True],
            [None],
            gamma=0.9,
            gae_lambda=0.95,
            last_value=99.0,
        )
        self.assertAlmostEqual(advantages[0], -1.0)
        self.assertAlmostEqual(returns[0], 1.0)

    def test_truncation_bootstraps_without_crossing_episode_boundary(self) -> None:
        advantages, returns = compute_gae(
            [1.0, 10.0],
            [2.0, 0.0],
            [False, False],
            [True, False],
            [3.0, None],
            gamma=0.9,
            gae_lambda=0.95,
            last_value=0.0,
        )
        self.assertAlmostEqual(advantages[0], 1.7)
        self.assertAlmostEqual(returns[0], 3.7)
        self.assertAlmostEqual(advantages[1], 10.0)

    def test_truncation_requires_bootstrap_value(self) -> None:
        with self.assertRaisesRegex(ValueError, "bootstrap value"):
            compute_gae(
                [1.0],
                [0.0],
                [False],
                [True],
                [None],
                gamma=0.99,
                gae_lambda=0.95,
                last_value=0.0,
            )


class RunningMomentsContractTest(unittest.TestCase):
    def test_incremental_moments_match_direct_statistics(self) -> None:
        first = np.array([[1.0, 10.0], [3.0, 14.0]])
        second = np.array([[5.0, 18.0], [7.0, 22.0], [9.0, 26.0]])
        merged = merge_rms_states(
            [compute_batch_moments(first), compute_batch_moments(second)]
        )
        expected = np.concatenate([first, second], axis=0)

        self.assertIsNotNone(merged)
        assert merged is not None
        np.testing.assert_allclose(merged["mean"], expected.mean(axis=0))
        np.testing.assert_allclose(merged["var"], expected.var(axis=0))
        self.assertEqual(merged["count"], float(len(expected)))


if __name__ == "__main__":
    unittest.main()
