@torch.jit.script
def compute_reward(object_rot: torch.Tensor, goal_rot: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute the relative rotation between object_rot and goal_rot quaternions.
    relative_rot = torch.abs(so3_relative_angle(object_rot, goal_rot))

    # The quaternion distance metric has a range of [0, pi] radians. We normalize this using a temperature parameter.
    temp_orientation = torch.tensor(math.pi, device=object_rot.device)
    reward_orientation = torch.exp(-relative_rot / temp_orientation)

    # Calculate the total reward.
    total_reward = reward_orientation

    # Store individual reward components in a dictionary.
    reward_dict = {"reward_orientation": reward_orientation}

    return total_reward, reward_dict
