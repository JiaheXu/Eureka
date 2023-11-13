@torch.jit.script
def compute_reward(object_rot: torch.Tensor, goal_rot: torch.Tensor, fingertip_pos: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Constants
    orientation_reward_weight = 1.0
    distance_reward_weight = 0.1

    # Calculate orientation reward
    orientation_diff = torch.matmul(object_rot, goal_rot.t())[:, 0]
    orientation_reward = orientation_reward_weight * orientation_diff

    # Calculate fingertip-object distance reward
    fingertip_count = fingertip_pos.shape[1]
    dist_reward = torch.tensor(0.0, device = fingertip_pos.device)
    
    for i in range(fingertip_count):
        fingertip_pos_i = fingertip_pos[:, i, :]
        fingertip_object_dist = torch.norm(fingertip_pos_i - object_rot[:, :3], dim=1)
        dist_reward += torch.exp(-distance_reward_weight * fingertip_object_dist)
    
    dist_reward /= fingertip_count

    # Calculate the total reward
    total_reward = orientation_reward + dist_reward
    
    # Return total reward and individual reward components
    reward_dict = {
        "orientation_reward": orientation_reward,
        "distance_reward": dist_reward
    }
    return total_reward, reward_dict
