@torch.jit.script
def compute_reward_v5(object_rot: torch.Tensor, goal_rot: torch.Tensor, 
                      object_angvel: torch.Tensor, dof_force_tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    
    device = object_rot.device
    
    # Compute the rotation error between the object and the goal orientation
    rot_error = torch.abs(object_rot - goal_rot)
    rot_error = torch.min(rot_error, 2 - rot_error)  # handle angle wraparound
    rot_error_sum = torch.sum(rot_error, dim=1)
    
    # Incentivize rotation alignment
    orientation_reward_temp = torch.tensor(1.0, device=device)
    orientation_reward = torch.exp(-orientation_reward_temp * rot_error_sum)
    
    # Penalize large angular velocities
    angvel_reward_temp = torch.tensor(1.0, device=device)  # Increase the temperature parameter
    angvel_penalty = torch.sum(torch.square(object_angvel), dim=1)
    angvel_reward = torch.exp(-angvel_reward_temp * angvel_penalty)

    # Update force_reward computation (new approach)
    force_magnitude = torch.sum(torch.abs(dof_force_tensor), dim=1)
    acceptable_force = torch.tensor(150, device=device)  # Experiment with different values
    force_diff = torch.abs(force_magnitude - acceptable_force)
    
    # Normalize force_reward using exponential transformation
    force_reward_temp = torch.tensor(0.01, device=device)  # Experiment with different values
    force_reward = torch.exp(-force_reward_temp * force_diff)

    # Combine reward components
    reward = orientation_reward * angvel_reward * force_reward

    reward_components = {"orientation_reward": orientation_reward,
                         "angvel_reward": angvel_reward,
                         "force_reward": force_reward}

    return reward, reward_components
