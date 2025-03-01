import retro
import abc
import numpy as np
from backend.multiplayer_games import interface


class StableRetroWrapper(interface.MultiPlayerKeyboardMouseGameInterfaceAEC):
    """Wrapper for Stable Retro multiplayer games"""

    def __init__(
        self,
        game: str,
        players: int = 2,
        state: str = retro.State.DEFAULT,
        scenario: str = None,
    ):
        self.env = retro.make(game, state=state, scenario=scenario, players=players)
        self.players = players
        self.current_player_index = 0
        self.agent_ids = [f"player_{i}" for i in range(players)]
        self.env.reset()
        self.pending_actions = {player: None for player in self.agent_ids}

    @abc.abstractmethod
    def map_action(
        self, action: interface.ValidActions, player_id: interface.PlayerId
    ) -> np.ndarray:
        """Map the action to the game's action space"""

    @property
    def current_player(self) -> interface.PlayerId:
        return self.agent_ids[self.current_player_index]

    def valid_actions(self) -> interface.ValidActions:
        return self.env.action_space

    def reset(self, seed: int = 42) -> None:
        self.env.seed(seed)
        self.env.reset()
        self.current_player_index = 0
        self.pending_actions = {player: None for player in self.agent_ids}

    def step(self, actions: interface.ValidActions, player_id: interface.PlayerId):
        assert player_id == self.current_player
        self.pending_actions[player_id] = self.map_action(actions, player_id)

        if all(action is not None for action in self.pending_actions.values()):
            full_action = np.concatenate(
                [self.pending_actions[player] for player in self.agent_ids]
            )
            obs, rewards, done, trunc, info = self.env.step(full_action)
            self.pending_actions = {player: None for player in self.agent_ids}
            return done, rewards

        return False, None

    def observe(self, player_id: interface.PlayerId) -> interface.Observations:
        return self.env.get_screen()

    def close(self) -> None:
        self.env.close()

    @property
    def window_size(self) -> tuple[int, int]:
        return self.env.observation_space.shape[:2]

    def get_agents(self) -> list[interface.PlayerId]:
        return self.agent_ids
