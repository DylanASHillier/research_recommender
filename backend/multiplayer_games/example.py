import asyncio
import gymnasium as gym

# from gymnasium.wrappers import Monitor
from backend.multiplayer_games.gymnasium_compat import GymnasiumWrapper
from backend.multiplayer_games.interface import AsyncAECGameLoop, PygameWindow
from backend.multiplayer_games.keyboard import PygameKeyboardWrapper

# Define the environment name
ENV_NAME = "LunarLander-v3"

# Define the key to action mapping
KEY_TO_ACTION = {
    "NOOP": 0,
    "space": 2,
    "left": 3,
    "right": 1,
    # "n": 0,  # NOOP
    # "f": 1,  # FIRE
    # "w": 2,  # UP
    # "d": 3,  # RIGHT
    # "a": 4,  # LEFT
    # "s": 5,  # DOWN
    # "e": 6,  # UPRIGHT
    # "q": 7,  # UPLEFT
    # "c": 8,  # DOWNRIGHT
    # "z": 9,  # DOWNLEFT
    # "r": 10,  # UPFIRE
    # "t": 11,  # RIGHTFIRE
    # "g": 12,  # LEFTFIRE
    # "b": 13,  # DOWNFIRE
    # "y": 14,  # UPRIGHTFIRE
    # "u": 15,  # UPLEFTFIRE
    # "i": 16,  # DOWNRIGHTFIRE
    # "o": 17,  # DOWNLEFTFIRE
}


class CustomGymnasiumWrapper(GymnasiumWrapper):
    def __init__(self, env):
        super().__init__(env)

    def map_action(self, action: set[str]):
        if len(action) != 0:
            print(action)
        # return KEY_TO_ACTION.get(action, 0)  # Default to NOOP if key not found
        for key in action:
            if key in KEY_TO_ACTION:
                return KEY_TO_ACTION[key]
        return 0  # Default to NOOP if key not found

    @property
    def window_size(self):
        """Need to override for box2d environments"""
        return 600, 400


async def main():
    # Create the Gymnasium environment
    env = gym.make(ENV_NAME, render_mode="rgb_array")
    env.reset()
    # env = Monitor(env, "./video", force=True, video_callable=lambda episode_id: True)
    wrapped_env = CustomGymnasiumWrapper(env)

    frame_size = wrapped_env.window_size

    # Create the game loop
    game_loop = AsyncAECGameLoop(wrapped_env)

    # Create the Pygame window with the actual frame size
    pygame_window = PygameWindow(game_loop, player_id=0, frame_size=frame_size)

    # Initialize Pygame and start observing the game state
    pygame_window.init_pygame()
    input_stream = PygameKeyboardWrapper()
    game_loop.setup_player_input(0, input_stream)
    await asyncio.gather(
        game_loop.game_loop(),
        pygame_window.observe_game_state(),
    )


if __name__ == "__main__":
    asyncio.run(main())
