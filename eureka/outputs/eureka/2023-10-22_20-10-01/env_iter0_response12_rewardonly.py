@torch.jit.script
def compute_reward(
    object_rot: torch.Tensor, 
    goal_rot: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Cosine similarity between object rotation and target rotation
    angle_diff = torch.einsum("ij,ij->i", object_rot, goal_rot)
    angle_diff = torch.clamp(angle_diff, -1.0, 1.0)
    angle_error = torch.acos(angle_diff)
    
    # Temperature parameters
    temp1: float = 0.1

    # Reward components
    orientation_reward = torch.exp(-temp1 * angle_error)

    # Overall reward
    reward = orientation_reward

    # Components dictionary
    reward_components = {
        "orientation_reward": orientation_reward
    }

    return reward, reward_components
