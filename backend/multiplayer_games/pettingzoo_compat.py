"""Compatability Code with AEC"""

from backend.multiplayer_games import interface
import pettingzoo
import abc


class PettingZooWrapper(interface.MultiPlayerKeyboardMouseGameInterfaceAEC):
    """Wrapper for PettingZoo games"""

    def __init__(self, env: pettingzoo.AECEnv):
        self.env = env

    @abc.abstractmethod
    def map_action(self, action: interface.ValidActions) -> int:
        """Map the action to the game's action space"""

    @property
    def current_player(self) -> interface.PlayerId:
        return self.env.agent_selection

    def valid_actions(self) -> interface.ValidActions:
        return self.env.action_spaces[self.current_player]

    def reset(self, seed: int = 42) -> None:
        self.env.reset(seed)

    def step(self, actions: interface.ValidActions, player_id: interface.PlayerId):
        actions = self.map_action(actions)
        assert player_id == self.current_player
        self.env.step(actions)

    def observe(self, player_id: interface.PlayerId) -> interface.Observations:
        return self.env.observe(player_id)

    def close(self) -> None:
        self.env.close()

    @property
    def window_size(self) -> tuple[int, int]:
        """Return the window size"""
        return 600, 400

    def get_agents(self) -> list[interface.PlayerId]:
        return list(self.env.agents)
