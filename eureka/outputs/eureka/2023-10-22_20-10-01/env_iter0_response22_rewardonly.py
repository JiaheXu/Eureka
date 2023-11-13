@torch.jit.script
def compute_reward(object_rot: Tensor, goal_rot: Tensor, object_angvel: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
    # Constants
    rotation_reward_weight: float = 1.0
    angular_velocity_reward_weight: float = 0.1
    rotation_temp: float = 10.0
    angular_velocity_temp: float = 20.0

    # Calculate difference between object and goal rotations
    rel_object_goal_rot = quat_mul(object_rot, quat_conjugate(goal_rot))

    # Compute rotation difference reward, range: [0, 1]
    rotation_diff = 1.0 - torch.abs(rel_object_goal_rot[:, 3])
    rotation_reward = torch.exp(rotation_temp * (rotation_diff - 1.0))

    # Compute angular velocity reward, range: [0, 1]
    goal_angvel = torch.zeros_like(object_angvel)
    angvel_diff = torch.norm((object_angvel - goal_angvel), dim=1)
    angvel_reward = torch.exp(-angular_velocity_temp * angvel_diff)

    # Combine rewards
    total_reward = rotation_reward_weight * rotation_reward + angular_velocity_reward_weight * angvel_reward

    # Create dictionary with individual reward components
    reward_components = {
        "rotation_reward": rotation_reward,
        "angvel_reward": angvel_reward
    }

    return total_reward, reward_components
