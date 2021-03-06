import numpy as np

from rlgym.utils.reward_functions.common_rewards import EventReward, ConditionalRewardFunction
from rlgym.utils.gamestates import GameState, PlayerData


class TimeLeftEventReward(EventReward):
    def __init__(self, max_timesteps=225, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.timestep = 0
        self.max_timesteps = max_timesteps

    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        self.timestep = 0

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray, optional_data=None):
        self.timestep += 1

        old_values = self.last_registered_values[player.car_id]
        new_values = self._extract_values(player, state)

        diff_values = new_values - old_values
        diff_values[diff_values < 0] = 0  # We only care about increasing values

        # copy original weights
        weights = np.array(self.weights)
        # weight goal reward by the remaining timesteps
        weights[0] *= max(self.max_timesteps - self.timestep, 5)
        weights[2] *= max(self.max_timesteps - self.timestep, 5)
        reward = np.dot(weights, diff_values)

        self.last_registered_values[player.car_id] = new_values
        return reward

class RewardIfFacingBall(ConditionalRewardFunction):
    def condition(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        pos_diff = state.ball.position - player.car_data.position
        norm_pos_diff = pos_diff / np.linalg.norm(pos_diff)
        return float(np.dot(player.car_data.forward(), norm_pos_diff)) > 0.7