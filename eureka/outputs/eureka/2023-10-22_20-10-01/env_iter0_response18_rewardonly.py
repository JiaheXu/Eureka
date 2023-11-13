@torch.jit.script
def compute_reward(object_rot: torch.Tensor, goal_rot: torch.Tensor, object_ang_vel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute the distance between the object's current rotation and the goal rotation
    quat_diff = quat_mul(object_rot, quat_conjugate(goal_rot))
    angle_diff = torch.acos(torch.abs(torch.clamp(quat_diff[..., 0], min=-1., max=1.))) * 2
    rot_diff_reward = -angle_diff
    
    # Normalize the rotation difference reward to the range [-1, 0] using exponentiation with a temperature parameter
    rotation_temperature = torch.tensor(1.0, device=object_rot.device)
    normalized_rot_diff_reward = torch.exp(rotation_temperature * rot_diff_reward) - 1
    
    # Reward for maintaining the desired angular velocity
    desired_ang_vel = torch.tensor(1.0, device=object_rot.device)
    ang_vel_diff = torch.abs(object_ang_vel - desired_ang_vel)
    
    # Normalize the angular velocity reward using exponentiation with a temperature parameter
    ang_vel_temperature = torch.tensor(1.0, device=object_rot.device)
    ang_vel_reward = torch.exp(-ang_vel_temperature * ang_vel_diff) - 1
    
    # Combine the rewards
    total_reward = normalized_rot_diff_reward + ang_vel_reward

    reward_components = {
        'normalized_rot_diff_reward': normalized_rot_diff_reward,
        'ang_vel_reward': ang_vel_reward
    }

    return total_reward, reward_components
