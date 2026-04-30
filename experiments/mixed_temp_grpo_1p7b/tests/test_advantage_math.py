import numpy as np
import pytest
import torch

pytest.importorskip("ray")

from experiments.mixed_temp_grpo_1p7b.trainer.mixed_advantage import compute_mixed_grouped_advantages


def test_mixed_advantage_zeroes_control_arm_by_default():
    token_level_rewards = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 2.0],
        ],
        dtype=torch.float32,
    )
    response_mask = torch.ones_like(token_level_rewards)
    uid = np.asarray(["g0", "g0"], dtype=object)
    arm = np.asarray(["control", "explore"], dtype=object)

    advantages, returns, diag = compute_mixed_grouped_advantages(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        uid=uid,
        arm=arm,
        update_control_arm=False,
        norm_adv_by_std=True,
    )

    expected = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(advantages, expected, atol=1e-5)
    assert torch.allclose(returns, expected, atol=1e-5)
    assert diag["reward/control_mean"] == 0.0
    assert diag["reward/explore_mean"] == 2.0
    assert diag["reward/gap"] == 2.0
