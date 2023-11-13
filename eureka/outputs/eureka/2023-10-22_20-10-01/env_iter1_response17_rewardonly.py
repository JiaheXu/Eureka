@torch.jit.script
def compute_reward(object_rot: torch.Tensor, goal_rot: torch.Tensor, 
                   object_angvel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    
    device = object_rot.device
    
    # Compute the relative rotation between the object and the goal orientation
    relative_rot = quat_relative_rotation(object_rot, goal_rot)
    
    # Incentivize rotation alignment
    orientation_reward_temp = torch.tensor(10.0, device=device)
    orientation_reward = torch.exp(-orientation_reward_temp * relative_rot)
    
    # Penalize large angular velocities
    angvel_reward_temp = torch.tensor(0.01, device=device)
    angvel_penalty = torch.sum(torch.square(object_angvel), dim=1)
    angvel_reward = torch.exp(-angvel_reward_temp * angvel_penalty)

    # Combine reward components
    reward = orientation_reward * angvel_reward

    reward_components = {"orientation_reward": orientation_reward,
                         "angvel_reward": angvel_reward}

    return reward, reward_components
