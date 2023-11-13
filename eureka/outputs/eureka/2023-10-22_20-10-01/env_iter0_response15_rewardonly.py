@torch.jit.script
def compute_reward(object_rot: torch.Tensor, goal_rot: torch.Tensor, fingertip_pos: torch.Tensor, grip_temp: float = 0.5, target_temp: float = 1.0) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

    # Calculate distance between current object rotation and goal rotation
    rotation_diff = quat_mul(object_rot, quat_conjugate(goal_rot))
    rotation_distance = torch.sum((rotation_diff - goal_rot) ** 2, dim=-1)

    # Calculate grip stability reward component
    fingertip_distances = torch.norm(fingertip_pos - object_rot[:, :3].unsqueeze(1), dim=-1)
    grip_stability = torch.mean(fingertip_distances, dim=1)
    grip_reward = torch.exp(-grip_stability / grip_temp)

    # Calculate target attainment reward component
    target_reward = torch.exp(-rotation_distance / target_temp)

    # Combine and balance reward components
    reward = 0.8 * target_reward + 0.2 * grip_reward

    reward_components = {"grip_reward": grip_reward, "target_reward": target_reward}

    return reward, reward_components
