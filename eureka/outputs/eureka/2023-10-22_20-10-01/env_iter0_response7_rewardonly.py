@torch.jit.script
def compute_reward(object_rot: torch.Tensor, goal_rot: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Calculate the quaternion distance between the current object rotation and the target goal rotation
    object_goal_rot_diff = quat_dist(object_rot, goal_rot)

    # Normalize the rotation difference to a fixed range using torch.exp and introduce a temperature parameter
    temperature_rot_diff = 1.0
    reward_rot_diff = -torch.exp(object_goal_rot_diff / temperature_rot_diff)
    
    # Calculate total reward
    total_reward = reward_rot_diff

    # Return total reward and individual reward components
    reward_components = {
        "reward_rot_diff": reward_rot_diff
    }

    return total_reward, reward_components
