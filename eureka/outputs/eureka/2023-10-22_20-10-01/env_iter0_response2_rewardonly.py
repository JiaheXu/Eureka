@torch.jit.script
def compute_reward(object_rot: torch.Tensor, goal_rot: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

    # Normalize quaternions to make sure they are valid
    object_rot_normalized = normalize_quat(object_rot)
    goal_rot_normalized = normalize_quat(goal_rot)

    # Calculate the absolute difference between object and goal rotations
    rot_diff = quat_mul(object_rot_normalized, quat_conjugate(goal_rot_normalized))
    angle_diff = 2.0 * torch.atan2(torch.norm(rot_diff[:, 1:4], dim=1), rot_diff[:, 0])

    # Set the temperature (hyperparameter) for the reward transformation
    temperature = torch.tensor(10.0, device=object_rot.device)

    # Calculate the orientation reward using exponential transformation
    orientation_reward = torch.exp(-temperature * angle_diff)

    # Return the total reward and a dictionary with individual reward components
    return orientation_reward, {"orientation_reward": orientation_reward}
