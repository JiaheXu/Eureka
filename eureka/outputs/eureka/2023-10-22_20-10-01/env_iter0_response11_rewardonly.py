@torch.jit.script
def compute_reward(object_rot: Tensor, goal_rot: Tensor, object_angvel: Tensor, goal_angvel: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
    
    # Constants
    rotation_reward_temperature: float = 10.0
    angvel_reward_temperature: float = 1.0

    # Compute rotation error (distance between current and target rotations)
    object_rot_inv = torch.tensor([
        object_rot[:, 0], -object_rot[:, 1], -object_rot[:, 2], -object_rot[:, 3]
    ]).t()
    
    rotation_error = torch.diagonal(
        torch.matmul(goal_rot, object_rot_inv), dim1=1, dim2=2
    )
    rotation_error = torch.clamp(rotation_error, min=-1.0, max=1.0)  

    # Compute angular velocity error
    angvel_error = torch.norm(object_angvel - goal_angvel, dim=1)

    # Calculate reward components for rotation and angular velocity
    rotation_reward = torch.exp(rotation_reward_temperature * (rotation_error - 1)) - 1
    angvel_reward = -torch.exp(angvel_reward_temperature * angvel_error) + 1

    # Calculate total reward
    total_reward = rotation_reward * angvel_reward

    reward_info = {
        "rotation_reward": rotation_reward,
        "angvel_reward": angvel_reward,
    }
    
    return total_reward, reward_info
