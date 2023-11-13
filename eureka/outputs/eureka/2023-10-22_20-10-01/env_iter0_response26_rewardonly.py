@torch.jit.script
def compute_reward(object_rot: torch.Tensor, goal_rot: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Constants
    rot_dist_temp = 10.0
    
    # Calculate the difference between the object's current and desired rotations
    rot_difference = torch.abs(torch.atan2(2 * (object_rot[:, 0] * object_rot[:, 3] + object_rot[:, 1] * object_rot[:, 2]),
                                          1 - 2 * (object_rot[:, 2] * object_rot[:, 2] + object_rot[:, 3] * object_rot[:, 3])) -
                              torch.atan2(2 * (goal_rot[:, 0] * goal_rot[:, 3] + goal_rot[:, 1] * goal_rot[:, 2]),
                                          1 - 2 * (goal_rot[:, 2] * goal_rot[:, 2] + goal_rot[:, 3] * goal_rot[:, 3])))

    # Normalize the rotation difference to the range [0, 1] and apply an exponential transformation
    normalized_rot_difference = rot_difference / (2 * torch.tensor(torch.pi, device=object_rot.device))
    rot_dist_reward = torch.exp(-rot_dist_temp * normalized_rot_difference)
    
    # Total reward
    total_reward = rot_dist_reward
    
    # Log the reward components
    reward_dict = {
        "rot_dist_reward": rot_dist_reward
    }
    
    return total_reward, reward_dict
