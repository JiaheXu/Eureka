@torch.jit.script
def compute_reward(object_rot: torch.Tensor, goal_rot: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute the error in object orientation in the form of an angle between the current and goal rotations
    q_diff = torch.mul(object_rot, torch.conj(goal_rot))
    angle_error = 2 * torch.atan2(torch.norm(q_diff[:, 1:], dim=1), torch.abs(q_diff[:, 0]))
    
    # Reward for matching the target orientation
    orientation_reward_temperature = 1.0
    orientation_reward = torch.exp(-orientation_reward_temperature * angle_error)
    
    # Combine the reward components
    total_reward = orientation_reward
    
    reward_components = {"orientation_reward": orientation_reward}
    
    return total_reward, reward_components
