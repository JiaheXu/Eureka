@torch.jit.script
def compute_reward(object_rot: torch.Tensor, goal_rot: torch.Tensor, 
                   object_angvel: torch.Tensor, dof_force_tensor: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    
    device = object_rot.device
    
    # Compute the rotation error between the object and the goal orientation
    rot_error = torch.abs(object_rot - goal_rot)
    rot_error = torch.min(rot_error, 2 - rot_error)  # handle angle wraparound
    rot_error_sum = torch.sum(rot_error, dim=1)
    
    # Incentivize rotation alignment
    orientation_reward_temp = torch.tensor(1.0, device=device)
    orientation_reward = torch.exp(-orientation_reward_temp * rot_error_sum)
    
    # Penalize large angular velocities
    angvel_reward_temp = torch.tensor(0.01, device=device)
    angvel_penalty = torch.sum(torch.square(object_angvel), dim=1)
    angvel_reward = torch.exp(-angvel_reward_temp * angvel_penalty)

    # Penalize large forces applied by the hand and large action magnitudes
    force_reward_temp = torch.tensor(0.001, device=device)
    force_penalty = torch.sum(torch.square(dof_force_tensor), dim=1)
    action_penalty = torch.sum(torch.square(actions), dim=1)
    violation_penalty = force_penalty + action_penalty
    force_reward = torch.exp(-force_reward_temp * violation_penalty)

    # Combine reward components
    reward = orientation_reward * angvel_reward * force_reward

    reward_components = {"orientation_reward": orientation_reward,
                         "angvel_reward": angvel_reward,
                         "force_reward": force_reward}

    return reward, reward_components
