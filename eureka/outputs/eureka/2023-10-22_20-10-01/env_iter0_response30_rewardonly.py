@torch.jit.script
def compute_reward(object_rot: torch.Tensor, goal_rot: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Calculate rotation difference (angular distance) between the object's current rotation and the target rotation
    rot_difference = torch.sum(object_rot * goal_rot, dim=1)
    angular_distance = 2.0 * torch.acos(torch.clamp(rot_difference, min=-1.0, max=1.0))  # Clamp to avoid numerical issues

    # Reward for reaching the target orientation
    orientation_temperature = 0.1
    orientation_reward = torch.exp(-1 * angular_distance / orientation_temperature)
    
    # Calculate total reward
    reward = orientation_reward

    # Store the individual components of the reward in a dictionary
    reward_components = {
        "orientation_reward": orientation_reward
    }

    return reward, reward_components
