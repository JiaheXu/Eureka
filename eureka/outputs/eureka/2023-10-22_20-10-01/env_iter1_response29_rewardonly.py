@torch.jit.script
def compute_reward_v2(object_rot: torch.Tensor, goal_rot: torch.Tensor, 
                      object_angvel: torch.Tensor, dof_force_tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    
    device = object_rot.device
    
    # Compute the rotation error between the object and the goal orientation
    rot_error = torch.abs(object_rot - goal_rot)
    rot_error = torch.min(rot_error, 2 - rot_error)  # handle angle wraparound
    rot_error_sum = torch.sum(rot_error, dim=1)
    
    # Incentivize rotation alignment
    orientation_reward_temp = torch.tensor(2.0, device=device)
    orientation_reward = torch.exp(-orientation_reward_temp * rot_error_sum)
    
    # Penalize large angular velocities
    angvel_reward_temp = torch.tensor(0.005, device=device)
    angvel_penalty = torch.sum(torch.square(object_angvel), dim=1)
    angvel_reward = torch.exp(-angvel_reward_temp * angvel_penalty)

    # Penalize large force applied by the hand
    force_reward_temp = torch.tensor(0.0001, device=device)
    force_penalty = torch.sum(torch.square(dof_force_tensor), dim=1)
    force_reward = torch.exp(-force_reward_temp * force_penalty)
    
    # Check if the force_reward affects the learned policy
    force_reward_threshold = torch.tensor(0.95, device=device)
    force_reward_check = (force_reward < force_reward_threshold).float()

    # Only apply force_reward when it is below the threshold
    combined_force_reward = torch.where(force_reward_check > 0, force_reward, torch.tensor(1.0, device=device))

    # Combine reward components
    reward = orientation_reward * angvel_reward * combined_force_reward

    reward_components = {"orientation_reward": orientation_reward,
                         "angvel_reward": angvel_reward,
                         "force_reward": combined_force_reward}

    return reward, reward_components
