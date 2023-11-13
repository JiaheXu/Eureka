@torch.jit.script
def compute_reward(object_rot: torch.Tensor, goal_rot: torch.Tensor, fingertip_pos: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Reward components:
    # - Reward function based on how well the object rotation matches the goal rotation
    # - Penalize the distance between fingertips and object
    
    # Calculate quaternion difference between current object rotation and the goal rotation
    quat_diff = quat_mul(object_rot, quat_conjugate(goal_rot))

    # Compute angle error using formula acos(2 * q0^2 - 1) where q0 is the scalar part of quaternion
    angle_error = torch.abs(torch.acos(2.0 * torch.square(quat_diff[:, 0]) - 1.0))

    # Calculate reward for matching object rotation with goal rotation
    rotation_reward = -angle_error

    # Penalize the distance between fingertips and object
    object_pos = quat_apply(object_rot, torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=fingertip_pos.device))
    dist_fingertips_object = torch.norm(fingertip_pos - object_pos, dim=-1).mean(dim=-1)
    distance_penalty = -dist_fingertips_object

    # Weights for the reward components
    w_rot = torch.tensor(1.0, dtype=torch.float32, device=object_rot.device)
    w_dist = torch.tensor(0.1, dtype=torch.float32, device=fingertip_pos.device)

    # Calculate total reward
    total_reward = w_rot * rotation_reward + w_dist * distance_penalty

    # Return total reward and individual reward components as a dictionary
    reward_components = {
        'rotation_reward': rotation_reward,
        'distance_penalty': distance_penalty
    }

    return total_reward, reward_components
