@torch.jit.script
def compute_reward(object_rot: torch.Tensor, goal_rot: torch.Tensor, object_angvel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Define temperature parameters
    orientation_temp: float = 10.0
    angvel_temp: float = 5.0

    # Compute distance between object's current rotation and target rotation
    rotation_distance = quat_distance(object_rot, goal_rot)

    # Calculate object's angular velocity direction
    object_angvel_normalized = torch.nn.functional.normalize(object_angvel, dim=1)

    # Calculate target angular velocity direction
    target_angvel = quat_mul(goal_rot, quat_conjugate(object_rot))
    target_angvel_normalized = torch.nn.functional.normalize(target_angvel[:, 1:], dim=1)

    # Compute cosine of angle between object's angular velocity and target angular velocity
    velocity_cosine_similarity = torch.sum(object_angvel_normalized * target_angvel_normalized, dim=1)

    # Normalize reward components with temperature parameters
    orientation_reward = torch.exp(-orientation_temp * rotation_distance)
    angvel_reward = torch.exp(angvel_temp * velocity_cosine_similarity)

    # Combine the reward components
    total_reward = orientation_reward + angvel_reward

    # Store the reward components in a dictionary
    reward_components = {"orientation_reward": orientation_reward, "angvel_reward": angvel_reward}

    return total_reward, reward_components
