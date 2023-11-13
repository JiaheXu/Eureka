@torch.jit.script
def compute_reward(object_rot: torch.Tensor, goal_rot: torch.Tensor,
                   object_angvel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Calculate the angular distance between the object's current rotation and the target rotation
    ang_dist = torch.acos(torch.abs(torch.bmm(goal_rot.view(-1,1,4), object_rot.view(-1,4,1)))) * 2
    
    # Reward for achieving the target orientation
    target_orientation_reward_weight = 1.0
    target_orientation_temperature = torch.tensor(0.2).to(ang_dist.device)
    target_orientation_reward = torch.exp(-ang_dist / target_orientation_temperature)
    
    # Reward and penalty based on angular velocity
    ang_vel_reward_weight = 0.1
    ang_vel_penalty_weight = 1.0
    ang_vel_threshold = torch.tensor(0.5).to(object_angvel.device)
    ang_vel_temperature = torch.tensor(3.0).to(object_angvel.device)

    angvel_norm = torch.norm(object_angvel, dim=1)
    ang_vel_reward = torch.exp(-angvel_norm / ang_vel_temperature)
    ang_vel_penalty = torch.where(angvel_norm > ang_vel_threshold, torch.tensor(1.0).to(ang_vel_penalty.device), torch.tensor(0.0).to(ang_vel_penalty.device))

    # Combine rewards and penalties
    reward = (target_orientation_reward_weight * target_orientation_reward
              + ang_vel_reward_weight * ang_vel_reward
              - ang_vel_penalty_weight * ang_vel_penalty)

    reward_info = {
        "target_orientation_reward": target_orientation_reward,
        "ang_vel_reward": ang_vel_reward,
        "ang_vel_penalty": ang_vel_penalty,
    }

    return reward, reward_info
