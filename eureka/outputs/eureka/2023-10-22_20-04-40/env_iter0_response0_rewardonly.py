@torch.jit.script
def compute_reward(object_rot: torch.Tensor, goal_rot: torch.Tensor,
                   object_angvel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Calculate orientation error between object's rotation and goal rotation
    rot_diff = torch.mul(object_rot, torch.conj(goal_rot))
    ori_error = 1.0 - torch.abs(torch.sum(rot_diff, dim=-1))

    # Reward for aligning the object's orientation with the target orientation
    orientation_reward_temperature = 5.0
    orientation_reward = torch.exp(-orientation_reward_temperature * ori_error)

    # Encourage spinning the object to reach the target orientation
    spin_reward_temperature = 5.0
    spin_reward = torch.exp(spin_reward_temperature * torch.norm(object_angvel, dim=-1))

    # Combine rewards and normalize to a range of [0, 1]
    total_reward = 0.5 * (orientation_reward + spin_reward)

    reward_components = {
        "orientation_reward": orientation_reward,
        "spin_reward": spin_reward,
    }

    return total_reward, reward_components
