@torch.jit.script
def compute_reward(object_rot: torch.Tensor, goal_rot: torch.Tensor, 
                   object_angvel: torch.Tensor, dof_force_tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    
    device = object_rot.device
    
    # Compute the rotation error between the object and the goal orientation
    rot_error = torch.abs(object_rot - goal_rot)
    rot_error = torch.min(rot_error, 2 - rot_error)  # handle angle wraparound
    rot_error_sum = torch.sum(rot_error, dim=1)
    
    # Incentivize rotation alignment
    orientation_reward_temp = torch.tensor(1.0, device=device)
    orientation_reward = torch.exp(-orientation_reward_temp * rot_error_sum)

    # Compute the alignment of object's angular velocities with the remaining rotation error
    angvel_alignment = torch.sum(object_angvel * rot_error, dim=1)

    # Incentivize angular velocity alignment with the rotation error
    angvel_alignment_temp = torch.tensor(1.0, device=device)
    angvel_alignment_reward = torch.exp(angvel_alignment_temp * angvel_alignment)
    
    # Combine reward components
    reward = orientation_reward * angvel_alignment_reward

    reward_components = {"orientation_reward": orientation_reward,
                         "angvel_alignment_reward": angvel_alignment_reward}

    return reward, reward_components
