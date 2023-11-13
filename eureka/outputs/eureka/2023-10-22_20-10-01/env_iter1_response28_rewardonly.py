@torch.jit.script
def compute_reward(object_rot: torch.Tensor, goal_rot: torch.Tensor,
                   object_angvel: torch.Tensor, dof_force_tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    
    device = object_rot.device

    # Compute the angular distance between the object and the goal orientation
    ang_distance = quat_angle_diff(object_rot, goal_rot)

    # Incentivize rotation alignment
    orientation_reward_temp = torch.tensor(5.0, device=device)
    orientation_reward = torch.exp(-orientation_reward_temp * ang_distance)

    # Penalize large angular velocities
    angvel_reward_temp = torch.tensor(0.01, device=device)
    angvel_penalty = torch.sum(torch.square(object_angvel), dim=1)
    angvel_reward = torch.exp(-angvel_reward_temp * angvel_penalty)

    # Penalize large force applied by the hand
    force_reward_temp = torch.tensor(0.0001, device=device)
    force_penalty = torch.sum(torch.square(dof_force_tensor), dim=1)
    force_reward = torch.exp(-force_reward_temp * force_penalty)

    # Combine reward components
    reward = orientation_reward * angvel_reward * force_reward

    reward_components = {"orientation_reward": orientation_reward,
                         "angvel_reward": angvel_reward,
                         "force_reward": force_reward}

    return reward, reward_components
