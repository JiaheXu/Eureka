@torch.jit.script
def compute_reward(object_rot: torch.Tensor, goal_rot: torch.Tensor, orientation_threshold: float = 0.1) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    orientation_reward_temp = 1.0

    # Calculate the orientation error between the object and the goal
    object_goal_rot_diff = torch.matmul(object_rot, torch.inverse(goal_rot))
    orientation_error = 1.0 - torch.abs(torch.diagonal(object_goal_rot_diff, dim1=-1, dim2=-2)).sum(dim=-1) / 4
    orientation_error = torch.clamp(orientation_error, 0, orientation_threshold)

    # Normalize the orientation error (range: 0-1)
    normalized_orientation_error = orientation_error / orientation_threshold

    # Calculate the orientation reward
    orientation_reward = torch.exp(-orientation_reward_temp * normalized_orientation_error)

    # Combine the reward components
    total_reward = orientation_reward

    # Pack reward components into a dictionary
    reward_info = {'orientation_reward': orientation_reward, 'total_reward': total_reward}

    return total_reward, reward_info
