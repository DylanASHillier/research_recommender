"""Wrapper for Gymnasium Games. If they support multiplayer
We require some way of breaking the action space into multiple players"""

from backend.multiplayer_games import interface


class GymnasiumWrapper(interface.MultiPlayerKeyboardMouseGameInterfaceAEC):
    """Wrapper for Gymnasium games"""

    def __init__(self, env):
        self.env = env

    @property
    def current_player(self):
        return 0

    @property
    def window_size(self):
        if len(self.env.observation_space.shape) == 2:
            return (
                self.env.observation_space.shape[1],
                self.env.observation_space.shape[0],
            )
        else:
            raise NotImplementedError("Only 2D observation spaces supported")

    def map_action(self, action: set[str]) -> int:
        raise NotImplementedError("Need to implement action mapping")

    def valid_actions(self):
        return self.env.action_space, False

    def reset(self, seed: int = 42):
        return self.env.reset(seed=seed)

    def step(self, actions, player_id):
        actions = self.map_action(actions)
        return self.env.step(actions)

    def observe(self, player_id):
        return self.env.render()

    def close(self):
        self.env.close()

    def get_agents(self):
        return [0]
