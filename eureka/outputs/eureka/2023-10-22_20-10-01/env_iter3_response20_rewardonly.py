@torch.jit.script
def compute_reward_v4(object_rot: torch.Tensor, goal_rot: torch.Tensor, 
                      object_angvel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    
    device = object_rot.device
    
    # Compute the rotation error between the object and the goal orientation
    rot_error = torch.abs(object_rot - goal_rot)
    rot_error = torch.min(rot_error, 2 - rot_error)  # handle angle wraparound
    rot_error_sum = torch.sum(rot_error, dim=1)
    
    # Incentivize rotation alignment
    orientation_reward_temp = torch.tensor(1.0, device=device)
    orientation_reward = torch.exp(-orientation_reward_temp * rot_error_sum)
    
    # Penalize large angular velocities
    angvel_reward_temp = torch.tensor(0.1, device=device)
    angvel_penalty = torch.sum(torch.square(object_angvel), dim=1)
    angvel_reward = torch.exp(-angvel_reward_temp * angvel_penalty)
    
    # Incentivize orientation-aligned angular velocity 
    orientation_velocity_reward_temp = torch.tensor(0.1, device=device)
    angular_alignment = torch.sum(torch.mul(object_rot, goal_rot), dim=1)
    orientation_velocity_reward = torch.exp(orientation_velocity_reward_temp * angular_alignment)
    
    # Combine reward components
    reward = orientation_reward * angvel_reward * orientation_velocity_reward

    reward_components = {"orientation_reward": orientation_reward,
                         "angvel_reward": angvel_reward,
                         "orientation_velocity_reward": orientation_velocity_reward}

    return reward, reward_components
