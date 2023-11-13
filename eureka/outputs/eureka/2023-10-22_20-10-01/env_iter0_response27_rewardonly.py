@torch.jit.script
def compute_reward(object_rot: torch.Tensor, goal_rot: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Constants
    rot_threshold = 0.005
    min_reward = -1.0
    max_reward = 1.0
    
    # Compute quaternion distance
    q_dist = torch.abs(torch.sum(torch.mul(goal_rot, object_rot), dim=1))
    rotation_reward = 1.0 - 2.0 * torch.pow(q_dist, 2)
    
    # Transform the rotation reward with a temperature parameter
    temperature_rot = torch.tensor(2.0, device=object_rot.device)
    trans_rotation_reward = torch.exp(temperature_rot * rotation_reward)
    
    # Normalize the transformed reward
    perceptual_reward = (trans_rotation_reward - torch.exp(temperature_rot * min_reward)) / (torch.exp(temperature_rot * max_reward) - torch.exp(temperature_rot * min_reward))
    
    # Ensure the reward is in the range [0, 1]
    reward = torch.clamp(perceptual_reward, 0.0, 1.0)
    
    # Calculate the individual reward components
    reward_components = {"RotationReward": rotation_reward, "TransformedRotationReward": trans_rotation_reward, "PerceptualReward": perceptual_reward}
    
    return reward, reward_components
