@torch.jit.script
def compute_reward(object_rot: torch.Tensor, goal_rot: torch.Tensor, 
                   object_angvel: torch.Tensor, dof_force_tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

    device = object_rot.device

    # Compute the rotation error between the object and the goal orientation
    rot_error = torch.abs(object_rot - goal_rot)
    rot_error = torch.min(rot_error, 2 - rot_error)  # handle angle wraparound
    rot_error_sum = torch.sum(rot_error, dim=1)

    # Incentivize rotation alignment
    orientation_reward_temp = torch.tensor(1.0, device=device)
    orientation_reward = torch.exp(-orientation_reward_temp * rot_error_sum)

    # Penalize large angular velocities
    angvel_reward_temp = torch.tensor(0.01, device=device)
    angvel_penalty = torch.sum(torch.square(object_angvel), dim=1)
    angvel_reward = torch.exp(-angvel_reward_temp * angvel_penalty)

    # Penalize large force applied by the hand
    force_reward_temp = torch.tensor(0.1, device=device)
    force_penalty = torch.sum(torch.square(dof_force_tensor), dim=1)
    force_reward = torch.exp(-force_reward_temp * force_penalty)

    # Incentivize proper rotational velocity
    rot_vel_desired = torch.tensor(2.0, device=device)
    rot_vel_reward_temp = torch.tensor(0.5, device=device)
    rot_vel_error = torch.abs(torch.norm(object_angvel, dim=1) - rot_vel_desired)
    rot_vel_reward = torch.exp(-rot_vel_reward_temp * rot_vel_error)

    # Combine reward components
    reward = orientation_reward * angvel_reward * force_reward * rot_vel_reward

    reward_components = {"orientation_reward": orientation_reward,
                         "angvel_reward": angvel_reward,
                         "force_reward": force_reward,
                         "rot_vel_reward": rot_vel_reward}

    return reward, reward_components
