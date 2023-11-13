@torch.jit.script
def compute_reward_v2(object_rot: torch.Tensor, goal_rot: torch.Tensor, 
                   object_angvel: torch.Tensor, dof_force_tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    
    device = object_rot.device
    
    # Compute the rotation error between the object and the goal orientation
    rot_error = torch.abs(object_rot - goal_rot)
    rot_error = torch.min(rot_error, 2 - rot_error)  # handle angle wraparound
    rot_error_sum = torch.sum(rot_error, dim=1)
    
    # Incentivize rotation alignment
    orientation_reward_temp = torch.tensor(2.0, device=device)  # Increased temperature from 1.0 to 2.0
    orientation_reward = torch.exp(-orientation_reward_temp * rot_error_sum)
    
    # Penalize large angular velocities
    angvel_reward_temp = torch.tensor(0.1, device=device)  # Increased temperature from 0.01 to 0.1
    angvel_penalty = torch.sum(torch.square(object_angvel), dim=1)
    angvel_reward = torch.exp(-angvel_reward_temp * angvel_penalty)

    # Penalize large force applied by the hand (re-scaled the penalty)
    force_reward_temp = torch.tensor(0.01, device=device)  # Increased temperature from 0.001 to 0.01
    force_penalty = torch.sum(torch.square(dof_force_tensor), dim=1) / 10000.0  # divided by a larger value for better scaling
    force_reward = torch.exp(-force_reward_temp * force_penalty)

    # Combine reward components
    reward = orientation_reward * angvel_reward * force_reward

    reward_components = {"orientation_reward": orientation_reward,
                         "angvel_reward": angvel_reward,
                         "force_reward": force_reward}

    return reward, reward_components
