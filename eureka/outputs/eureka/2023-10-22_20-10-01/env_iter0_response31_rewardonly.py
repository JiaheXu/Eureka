@torch.jit.script
def compute_reward(object_rot: torch.Tensor, goal_rot: torch.Tensor, fingertip_pos: torch.Tensor, object_pos: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute the quaternion distance between the current object rotation and the target rotation
    q1 = object_rot
    q2 = goal_rot
    
    dot_product = torch.sum(q1 * q2, dim=-1)
    angle_distance = 2 * torch.arccos(torch.abs(dot_product))
    
    # Compute a position reward for keeping the fingertips close to the object
    fingertip_distance = torch.norm(fingertip_pos - object_pos.unsqueeze(dim=1), dim=-1)
    fingertip_reward = torch.exp(-fingertip_distance)

    # Normalize the angle distance to be in the range [0, pi]
    normalized_angle_distance = angle_distance / torch.tensor(3.141592653589793).to(angle_distance.device)

    # Calculate the rotation reward using an exponential function
    rotation_reward_temperature = 10.0
    rotation_reward = torch.exp(-rotation_reward_temperature * normalized_angle_distance)
    
    # Assign weights to the reward components
    rotation_weight = 1.0
    fingertip_weight = 0.1

    # Compute the total reward as a weighted combination of rotation reward and fingertip reward
    total_reward = rotation_weight * rotation_reward + fingertip_weight * fingertip_reward

    # Return the total reward and the individual reward components
    rewards_dict = {"rotation_reward": rotation_reward, "fingertip_reward": fingertip_reward}
    return total_reward, rewards_dict
