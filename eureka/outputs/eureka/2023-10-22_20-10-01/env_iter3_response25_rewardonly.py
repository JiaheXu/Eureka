@torch.jit.script
def compute_reward_v4(object_rot: torch.Tensor, goal_rot: torch.Tensor, 
                      object_angvel: torch.Tensor, dof_force_tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    
    device = object_rot.device
    
    # Compute the rotation error between the object and the goal orientation
    rot_error = torch.abs(object_rot - goal_rot)
    rot_error = torch.min(rot_error, 2 - rot_error)  # handle angle wraparound
    rot_error_sum = torch.sum(rot_error, dim=1)
    
    # Incentivize rotation alignment (updated component)
    orientation_reward_temp = torch.tensor(2.0, device=device)  # increased temperature parameter
    orientation_reward = torch.exp(-orientation_reward_temp * rot_error_sum)
    
    # Penalize large angular velocities (updated component)
    angvel_reward_temp = torch.tensor(2.0, device=device)  # increased temperature parameter
    angvel_penalty = torch.sum(torch.square(object_angvel), dim=1)
    angvel_reward = torch.exp(-angvel_reward_temp * angvel_penalty)

    # Penalize force change rate (new component)
    force_change_reward_temp = torch.tensor(1.0, device=device)
    force_delta = dof_force_tensor[:, 1:] - dof_force_tensor[:, :-1]
    force_change_penalty = torch.mean(torch.sum(torch.square(force_delta), dim=-1), dim=-1)
    force_change_reward = torch.exp(-force_change_reward_temp * force_change_penalty)

    # Combine reward components
    reward = orientation_reward * angvel_reward * force_change_reward

    reward_components = {"orientation_reward": orientation_reward,
                         "angvel_reward": angvel_reward,
                         "force_change_reward": force_change_reward}

    return reward, reward_components
