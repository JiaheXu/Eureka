@torch.jit.script
def compute_reward(object_rot: torch.Tensor, goal_rot: torch.Tensor, fingertip_pos: torch.Tensor, object_pos: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Temperature parameters
    orientation_temp: float = 1.0
    fingers_dist_temp: float = 1.0

    # Calculate the orientation error
    object_rot_inv = torch.inverse(object_rot)
    goal_rot_relative = torch.matmul(goal_rot, object_rot_inv)
    trace = torch.diagonal(goal_rot_relative, dim1=1, dim2=2).sum(dim=1)
    orientation_error = (1.0 - trace) / 4.0
    orientation_reward = torch.exp(-orientation_temp * orientation_error)

    # Calculate the average distance between fingertips and the object
    object_pos_expanded = object_pos.unsqueeze(1).repeat(1, fingertip_pos.shape[1], 1)
    dist_fingers_object = torch.norm(fingertip_pos - object_pos_expanded, dim=2)
    avg_dist_fingers_object = dist_fingers_object.mean(dim=1)
    fingers_distance_reward = torch.exp(-fingers_dist_temp * avg_dist_fingers_object)

    # Calculate the total reward
    total_reward = orientation_reward * fingers_distance_reward

    # Individual reward components
    reward_components = { "orientation_reward": orientation_reward, "fingers_distance_reward": fingers_distance_reward }

    return total_reward, reward_components
