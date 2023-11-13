@torch.jit.script
def compute_reward(object_rot: torch.Tensor, goal_rot: torch.Tensor, fingertip_pos: torch.Tensor, object_pos: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    orientation_error = 1.0 - torch.matmul(object_rot, goal_rot.transpose(0, 1))  # (num_envs,)
    distance_to_object = torch.norm(fingertip_pos - object_pos.unsqueeze(1), dim=-1).mean(dim=-1)  # (num_envs,)
    
    orientation_temperature = 10.0
    object_temperature = 0.1

    orientation_reward = torch.exp(-orientation_error / orientation_temperature)  # (num_envs,)
    object_reward = torch.exp(-distance_to_object / object_temperature)  # (num_envs,)

    total_reward = orientation_reward * object_reward
    
    reward_dict = {
        'orientation_reward': orientation_reward,
        'object_reward': object_reward
    }

    return total_reward, reward_dict
