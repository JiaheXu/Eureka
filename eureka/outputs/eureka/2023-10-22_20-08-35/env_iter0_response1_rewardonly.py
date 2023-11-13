@torch.jit.script
def compute_reward(object_rot: torch.Tensor, goal_rot: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    orientation_err_temperature = 1.0
    rotation_err_temperature = 1.0
    
    # Compute the orientation error between the object and the target
    orientation_err_quat = torch.tensor([object_rot[:, 3], -object_rot[:, 0], -object_rot[:, 1], -object_rot[:, 2]], dtype=torch.float32)
    orientation_err_quat = orientation_err_quat.to(object_rot.device)
    orientation_err = torch.abs(torch.sum(object_rot * goal_rot, dim=1) - 1.0) * 0.5
    
    # Compute the reward based on the normalized orientation error
    orientation_reward = -torch.exp(orientation_err * orientation_err_temperature)
    
    # Compute the rotation error between the object and the target
    rotation_err_quat = torch.mul(orientation_err_quat, goal_rot)
    rotation_err = torch.abs(torch.atan2(
        2 * torch.sum(rotation_err_quat * torch.tensor([[0, -1, 2, 1]], dtype=torch.float32), dim=1),
        1 - 2 * torch.sum(torch.pow(rotation_err_quat[:, 1:], 2), dim=1)
    ))
    
    # Compute the reward based on the normalized rotation error
    rotation_reward = -torch.exp(rotation_err * rotation_err_temperature)
    
    total_reward = orientation_reward + rotation_reward
    
    return total_reward, {"orientation_reward": orientation_reward, "rotation_reward": rotation_reward}
