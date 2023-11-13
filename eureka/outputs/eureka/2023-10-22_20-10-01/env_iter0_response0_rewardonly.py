@torch.jit.script
def compute_reward(object_rot: torch.Tensor, goal_rot: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Parameters
    quat_distance_temp = 2.0

    # Compute quaternion distance between object_rot and goal_rot
    quat_diff = torch.mul(object_rot, torch.inverse(goal_rot))
    goal_reached = torch.abs(torch.sum(torch.mul(quat_diff, quat_diff), dim=1) - 2.0)

    # Apply exponential transformation to quat_distance
    quat_distance_reward = torch.exp(-quat_distance_temp * goal_reached)

    # Calculate the total reward
    total_reward = quat_distance_reward

    # Create a dictionary for the individual reward components
    reward_dict = {
        'quat_distance_reward': quat_distance_reward,
    }

    return total_reward, reward_dict
