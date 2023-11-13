@torch.jit.script
def compute_reward(object_rot: torch.Tensor, goal_rot: torch.Tensor, fingertip_pos: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute the rotation error between the object and the goal
    object_rot_inv = quat_conjugate(object_rot)
    rotation_error = quat_mul(object_rot_inv, goal_rot)
    angle_error = 2 * torch.atan2(torch.norm(rotation_error[:, 1:], dim=-1), rotation_error[:, 0])

    # Compute reward related to rotation error
    rotation_reward_weight = 1.0
    rotation_reward_temperature = 1.0
    rotation_reward = torch.exp(-rotation_reward_temperature * angle_error)
    weighted_rotation_reward = rotation_reward_weight * rotation_reward

    # Penalize large fingertip movements
    fingertip_movement_weight = -0.1
    fingertip_movement_temperature = 1.0
    fingertip_movement = torch.norm(fingertip_pos[:, 1:] - fingertip_pos[:, :-1], dim=-1).sum(-1)
    fingertip_movement_reward = torch.exp(-fingertip_movement_temperature * fingertip_movement)
    weighted_fingertip_movement_reward = fingertip_movement_weight * fingertip_movement_reward

    # Calculate total reward
    total_reward = weighted_rotation_reward + weighted_fingertip_movement_reward

    # Store the individual reward components into a dictionary
    reward_components = {
        'weighted_rotation_reward': weighted_rotation_reward,
        'weighted_fingertip_movement_reward': weighted_fingertip_movement_reward
    }

    return total_reward, reward_components
