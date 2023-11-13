@torch.jit.script
def compute_reward(object_rot: torch.Tensor, goal_rot: torch.Tensor, fingertip_pos: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Constants
    rot_term_temperature: float = 0.1
    contact_term_temperature: float = 10.0

    # Compute the rotation error between object rotation and goal rotation
    quat_diff = quat_mul(object_rot, quat_conjugate(goal_rot))
    rotation_error = torch.abs(torch.atan2(2 * quat_diff[:, 0] * quat_diff[:, 3] - 2 * quat_diff[:, 1] * quat_diff[:, 2], 1 - 2 * quat_diff[:, 0] * quat_diff[:, 0] - 2 * quat_diff[:, 2] * quat_diff[:, 2]))

    # Compute the contact term
    fingertips_min_distance = torch.min(torch.std(fingertip_pos, dim=1), dim=1).values
    contact_term = torch.exp(-contact_term_temperature * fingertips_min_distance)

    # Combine error terms into the total reward
    rotation_reward = torch.exp(-rot_term_temperature * rotation_error)
    total_reward = rotation_reward * contact_term

    # Create a dictionary for the reward components
    reward_components = {
        "rotation_reward": rotation_reward,
        "contact_term": contact_term
    }

    return total_reward, reward_components
