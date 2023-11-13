@torch.jit.script
def compute_reward_v3(object_rot: torch.Tensor, goal_rot: torch.Tensor, 
                      object_angvel: torch.Tensor, dof_force_tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    
    device = object_rot.device
    
    # Compute the relative quaternion error between the object and the goal orientation
    q_rel = quat_mul(goal_rot, quat_conjugate(object_rot))
    angle_error = 2 * torch.atan2(torch.norm(q_rel[:, 1:], dim=1), torch.abs(q_rel[:, 0]))  # angle error in radians
    angle_error = torch.min(angle_error, 2 * torch.tensor(torch.pi, device=device) - angle_error)  # handle angle wraparound
    
    # Incentivize rotation alignment
    orientation_reward_temp = torch.tensor(0.5, device=device)  # increased the temperature parameter
    orientation_reward = torch.exp(-orientation_reward_temp * angle_error)
    
    # Penalize large angular velocities
    angvel_reward_temp = torch.tensor(0.1, device=device)
    angvel_penalty = torch.sum(torch.square(object_angvel), dim=1)
    angvel_reward = torch.exp(-angvel_reward_temp * angvel_penalty)

    # Penalize large force applied by the hand (updated component)
    force_reward_temp = torch.tensor(0.01, device=device)  
    force_penalty = torch.mean(torch.square(dof_force_tensor), dim=1)
    force_reward = torch.exp(-force_reward_temp * force_penalty)

    # Combine reward components
    reward = orientation_reward * angvel_reward * force_reward

    reward_components = {"orientation_reward": orientation_reward,
                         "angvel_reward": angvel_reward,
                         "force_reward": force_reward}

    return reward, reward_components
