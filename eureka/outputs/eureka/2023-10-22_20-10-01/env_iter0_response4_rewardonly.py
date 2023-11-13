@torch.jit.script
def compute_reward(object_rot: torch.Tensor, goal_rot: torch.Tensor, object_angvel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    quat_diff_reward_temp = 1.0
    angvel_bonus_temp = 0.1

    rot_diff = quat_diff(object_rot, goal_rot)
    quat_diff_reward = torch.exp(quat_diff_reward_temp * (rot_diff - 1))

    angvel_norm = torch.norm(object_angvel)
    angvel_bonus = torch.exp(angvel_bonus_temp * (angvel_norm - 1))

    reward = quat_diff_reward + angvel_bonus
    reward_components = {"quat_diff_reward": quat_diff_reward, "angvel_bonus": angvel_bonus}

    return reward, reward_components
