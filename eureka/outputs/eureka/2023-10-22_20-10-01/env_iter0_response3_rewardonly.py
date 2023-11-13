@torch.jit.script
def compute_reward(object_rot: torch.Tensor, goal_rot: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute the quaternion difference between the object and target rotations
    quat_diff = quat_sub(object_rot, goal_rot)

    # Compute the L2 loss between the object and target quaternions
    orientation_loss = quat_l2_loss(object_rot, goal_rot)

    # Normalize the L2 loss with the temperature parameter
    orientation_temperature = torch.tensor(0.1, device=object_rot.device)
    orientation_reward = -torch.exp(orientation_loss / orientation_temperature)

    # The total reward is the orientation_reward
    total_reward = orientation_reward

    # Store the individual reward components in a dictionary
    reward_components = {'orientation_reward': orientation_reward}

    return total_reward, reward_components
