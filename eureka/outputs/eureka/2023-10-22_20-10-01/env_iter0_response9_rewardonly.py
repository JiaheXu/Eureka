@torch.jit.script
def compute_reward(object_rot: torch.Tensor, goal_rot: torch.Tensor, dof_force_tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute the distance between current and goal quaternions
    q_diff = quat_mul(object_rot, quat_conjugate(goal_rot))
    q_diff_angle = 2 * torch.atan2(torch.norm(q_diff[:, 1:], dim=1), q_diff[:, 0]).unsqueeze(1)

    # Reward function components
    rot_reward = torch.exp(-q_diff_angle)
    control_reward = -torch.mean(torch.abs(dof_force_tensor), dim=1).unsqueeze(1)

    # Temperature parameters for reward components
    temp_rot = 1.0
    temp_control = 0.1

    # Adjust rewards with their temperature parameters
    rot_reward_scaled = rot_reward / temp_rot
    control_reward_scaled = control_reward / temp_control

    # Total reward
    total_reward = rot_reward_scaled + control_reward_scaled

    # Pack individual reward components into a dictionary
    reward_info = {
        'rot_reward': rot_reward,
        'control_reward': control_reward,
    }

    return total_reward, reward_info
