@torch.jit.script
def compute_reward(object_rot: torch.Tensor, goal_rot: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Calculate the rotation error
    object_rot_inv = torch.tensor([object_rot[:, 0], -object_rot[:, 1], -object_rot[:, 2], -object_rot[:, 3]])
    rot_error = torch.abs(torch.sum(object_rot_inv * goal_rot, dim=1))

    # Linear temperature parameter for smoothing the reward
    temp_rot_error = 1.0

    # Apply transformation to scale values to an appropriate range
    rot_error_exp = torch.exp(-temp_rot_error * rot_error)

    # Calculate total reward
    reward = rot_error_exp

    # Pack individual reward components into a dictionary
    reward_info = {
        'rot_error_exp': rot_error_exp
    }

    return reward, reward_info
