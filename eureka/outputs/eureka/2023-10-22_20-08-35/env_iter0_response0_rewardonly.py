@torch.jit.script
def compute_reward(object_rot: torch.Tensor, goal_rot: torch.Tensor, fingertip_state: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute the orientation difference between object and goal rotations
    object_rot_inverse = quat_conjugate(object_rot)
    orientation_diff = quat_mul(goal_rot, object_rot_inverse)
    
    # Convert quaternion difference to angle
    angle_diff = 2 * torch.atan2(torch.norm(orientation_diff[:, 1:], dim=1), orientation_diff[:, 0]).unsqueeze(1)
    
    # Normalize the angle difference to range [0, 1]
    angle_diff_normalized = angle_diff / torch.tensor([3.14159265359]).to(object_rot.device)
    
    # Orientation reward (higher for smaller orientation differences)
    orientation_reward = torch.exp(-100 * angle_diff_normalized)

    # Fingertip proximity reward (higher for closer fingertips to the object)
    fingertip_proximity = torch.mean(torch.norm(fingertip_state[:, :, 0:3] - object_rot[:, None, 0:3], dim=2), dim=1).unsqueeze(1)
    proximity_reward = torch.exp(-10 * fingertip_proximity)
    
    # Total reward (weighted combination of orientation and proximity rewards)
    total_reward = 0.7 * orientation_reward + 0.3 * proximity_reward
    
    # Return the reward and its components as a dictionary
    reward_components = {
        "orientation_reward": orientation_reward,
        "proximity_reward": proximity_reward,
    }

    return total_reward, reward_components
