@torch.jit.script
def compute_reward(object_rot: torch.Tensor, goal_rot: torch.Tensor, fingertip_pos: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Calculate distance between object's current rotation and target rotation
    q_diff = quat_mul(goal_rot, quat_conjugate(object_rot))
    angle_diff = 2 * torch.atan2(torch.norm(q_diff[:, 1:], dim=1), q_diff[:, 0])
    rotation_distance = angle_diff / torch.tensor(3.14159265, device=angle_diff.device)
    
    # Calculate penalty for large fingertip distances to the object
    ref_pos = fingertip_pos.mean(dim=1)
    avg_fingertip_distance = torch.norm(fingertip_pos - ref_pos.unsqueeze(1), dim=2).mean(dim=1)
    fingertip_distance_penalty = torch.exp(avg_fingertip_distance) - 1.0
    
    # Compute reward components
    rotation_reward = torch.exp(-torch.tensor(1.0, device=rotation_distance.device) * rotation_distance)
    distance_penalty = -torch.tensor(0.1, device=fingertip_distance_penalty.device) * fingertip_distance_penalty
    total_reward = rotation_reward + distance_penalty
    
    # Store the individual reward components
    reward_dict = {
        'rotation_reward': rotation_reward,
        'distance_penalty': distance_penalty,
    }

    return total_reward, reward_dict
