@torch.jit.script
def compute_reward_v5(object_rot: torch.Tensor, goal_rot: torch.Tensor, 
                      object_angvel: torch.Tensor, dof_force_tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    
    device = object_rot.device
    
    # Compute the rotation error between the object and the goal orientation
    rot_error = torch.abs(object_rot - goal_rot)
    rot_error = torch.min(rot_error, 2 - rot_error)  # handle angle wraparound
    rot_error_sum = torch.sum(rot_error, dim=1)

    # Incentivize rotation alignment and rescale it (Updated component)
    orientation_reward_temp = torch.tensor(2.0, device=device)  # Updated value
    orientation_reward = torch.exp(-orientation_reward_temp * rot_error_sum)

    # Penalize large angular velocities and rescale it (Updated component)
    angvel_reward_temp = torch.tensor(1.0, device=device)  # Updated value
    angvel_penalty = torch.sum(torch.square(object_angvel), dim=1)
    angvel_reward = torch.exp(-angvel_reward_temp * angvel_penalty)

    # Compute the force magnitude and smoothen the force reward (Updated component)
    force_magnitude = torch.sum(torch.square(dof_force_tensor), dim=1)

    force_limit = torch.tensor(1000, device=device)  # tweak force_limit value to find optimal constraint
    force_reward_temp = torch.tensor(0.01, device=device)  # New temperature parameter
    force_penalty = torch.clamp(force_magnitude - force_limit, min=0)  # Penalize only when force_magnitude is greater than force_limit
    force_reward = torch.exp(-force_reward_temp * force_penalty)  

    # Combine reward components
    reward = 1/3 * (orientation_reward + angvel_reward + force_reward)

    reward_components = {"orientation_reward": orientation_reward,
                         "angvel_reward": angvel_reward,
                         "force_reward": force_reward}

    return reward, reward_components
