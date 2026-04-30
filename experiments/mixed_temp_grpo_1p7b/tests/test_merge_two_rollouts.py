import numpy as np
import pytest

from experiments.mixed_temp_grpo_1p7b.trainer.rollout_utils import build_group_uids, build_prompt_ids, repeat_prompt_metadata


def test_repeat_prompt_metadata_preserves_uid_grouping_across_two_arms():
    prompt_ids = np.asarray([11, 22], dtype=object)
    uid_base = build_group_uids(prompt_ids=prompt_ids, global_step=7)

    control_uid, _, control_arm, _, control_idx, control_steps = repeat_prompt_metadata(
        prompt_ids=prompt_ids,
        uid_base=uid_base,
        repeat_times=2,
        arm_name="control",
        temperature=0.3,
        global_step=7,
        sample_idx_start=0,
    )
    explore_uid, _, explore_arm, _, explore_idx, explore_steps = repeat_prompt_metadata(
        prompt_ids=prompt_ids,
        uid_base=uid_base,
        repeat_times=6,
        arm_name="explore",
        temperature=1.0,
        global_step=7,
        sample_idx_start=2,
    )

    merged_uid = np.concatenate([control_uid, explore_uid])
    merged_arm = np.concatenate([control_arm, explore_arm])
    merged_idx = np.concatenate([control_idx, explore_idx])
    merged_steps = np.concatenate([control_steps, explore_steps])

    for uid in uid_base:
        uid_mask = merged_uid == uid
        assert int(uid_mask.sum()) == 8
        assert int((merged_arm[uid_mask] == "control").sum()) == 2
        assert int((merged_arm[uid_mask] == "explore").sum()) == 6
        assert sorted(merged_idx[uid_mask].tolist()) == list(range(8))

    assert set(merged_steps.tolist()) == {7}


def test_build_prompt_ids_requires_stable_index_by_default():
    batch = type("FakeBatch", (), {"non_tensor_batch": {}})()
    with pytest.raises(ValueError):
        build_prompt_ids(batch)
