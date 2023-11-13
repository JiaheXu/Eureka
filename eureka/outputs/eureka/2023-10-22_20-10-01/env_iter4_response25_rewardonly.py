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
    angvel_reward_temp = torch.tensor(0.1, device=device)
    angvel_penalty = torch.sum(torch.square(object_angvel), dim=1)
    angvel_reward = torch.exp(-angvel_reward_temp * angvel_penalty)

    # Compute the force magnitude and introduce force_limit (updated component)
    force_magnitude = torch.sum(torch.square(dof_force_tensor), dim=1)
    force_limit = torch.tensor(50, device=device)  # reduce force_limit to 50, experiment with different values
    
    # Force_penalty updated to use sigmoid function
    force_penalty_temp = torch.tensor(0.01, device=device)
    force_penalty = 1 - torch.sigmoid(force_penalty_temp * (force_limit - force_magnitude))
    
    # Combine reward components (updated)
    force_reward = 1 - force_penalty
    reward = orientation_reward * angvel_reward * force_reward

    reward_components = {"orientation_reward": orientation_reward,
                         "angvel_reward": angvel_reward,
                         "force_reward": force_reward}

    # Uncomment this part during training to verify the uniqueness of each value in reward components
    #unique_orient, _ = torch.unique(orientation_reward, return_counts=True)
    #unique_angvel, _ = torch.unique(angvel_reward, return_counts=True)
    #unique_force, _ = torch.unique(force_reward, return_counts=True)
    #print("Unique Orientation Rewards:", unique_orient.cpu().numpy())
    #print("Unique Angular Velocity Rewards:", unique_angvel.cpu().numpy())
    #print("Unique Force Rewards:", unique_force.cpu().numpy())

    return reward, reward_components
