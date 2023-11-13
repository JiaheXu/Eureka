@torch.jit.script
def compute_reward(object_rot: torch.Tensor, goal_rot: torch.Tensor, fingertip_pos: torch.Tensor, object_pos: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Constants and temperature parameters
    goal_dist_thresh: float = 0.1  # meters
    reach_temp: float = -5.0
    rot_temp: float = -10.0

    # Distance from fingertips to the object position
    fingertip_dist = torch.norm(object_pos.view(1, 3) - fingertip_pos, dim=-1)
    dist_rew = torch.exp(reach_temp * fingertip_dist)

    # Compute the difference in rotation between object and goal
    object_inv_rot = torch.zeros_like(object_rot)
    object_inv_rot[..., 0] = object_rot[..., 0]
    object_inv_rot[..., 1:] = -object_rot[..., 1:]
    rot_difference = torch.matmul(goal_rot, object_inv_rot[..., None])[..., 0]

    # Compute the angle from rotation difference
    cos_half_angle = rot_difference[..., 0]
    angle = torch.acos(torch.clamp(cos_half_angle, min=-1.0, max=1.0))

    # Compute the rotation reward
    rot_rew = torch.exp(rot_temp * angle)

    # If the rotation reward is within the threshold, the agent gets additional reward
    in_thresh = (angle < goal_dist_thresh).type_as(angle)
    in_thresh_rew = in_thresh * 1.0
    total_rew = (dist_rew + rot_rew) + in_thresh_rew

    rew_components = {"dist_rew": dist_rew, "rot_rew": rot_rew, "in_thresh_rew": in_thresh_rew}
    return total_rew, rew_components
