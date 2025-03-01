import asyncio
import gymnasium as gym
from PIL import Image
from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration
from backend.multiplayer_games.gymnasium_compat import GymnasiumWrapper
from backend.multiplayer_games.interface import (
    AsyncAECGameLoop,
    PygameWindow,
    AsyncLLMWrapper,
    InputActionStream,
)
from backend.multiplayer_games import keyboard
from backend.multiplayer_games.keyboard import PygameKeyboardWrapper

# Define the environment name
ENV_NAME = "LunarLander-v3"

# Define the key to action mapping
KEY_TO_ACTION = {
    "NOOP": 0,
    "left": 1,
    "main engine": 2,
    "right": 3,
}


class CustomGymnasiumWrapper(GymnasiumWrapper):
    def __init__(self, env):
        super().__init__(env)

    def map_action(self, action: set[str]):
        for key in action:
            if key in KEY_TO_ACTION:
                return KEY_TO_ACTION[key]
        return 0  # Default to NOOP if key not found

    def window_size(self):
        """Need to override for box2d environments"""
        return 600, 400


LLMAction = str


class LLMActionStream(InputActionStream[LLMAction]):
    def __init__(self, use_buffer=False, buffer_size=100):
        super().__init__(use_buffer=use_buffer, buffer_size=buffer_size)

    def update_state(self, event: LLMAction):
        self.current_state = event


class LLMActionWrapper(AsyncLLMWrapper):
    def __init__(self, game_loop, player_id, input_stream):
        super().__init__(game_loop, player_id, input_stream)
        self.processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )

    async def compute_actions(self, observations) -> LLMAction:
        # Convert observation to image
        print(f"Computing actions over {len(observations)} observations")
        images = [Image.fromarray(observation) for observation in observations]
        # Prepare inputs for the model
        inputs = self.processor(images=images, return_tensors="pt")
        # Generate action using LLM
        loop = asyncio.get_event_loop()
        outputs = await loop.run_in_executor(
            None, lambda inputs: self.model.generate(**inputs), inputs
        )
        action_text = self.processor.decode(
            outputs[0], skip_special_tokens=True
        ).lower()
        print(action_text)
        # Map generated text to action
        if "left" in action_text:
            return "left"
        elif "right" in action_text:
            return "right"
        elif "main" in action_text or "engine" in action_text:
            return "main engine"
        else:
            return "NOOP"


async def main():
    # Create the Gymnasium environment
    env = gym.make(ENV_NAME, render_mode="rgb_array")
    env.reset()
    wrapped_env = CustomGymnasiumWrapper(env)

    frame_size = wrapped_env.window_size()

    # Create the game loop
    game_loop = AsyncAECGameLoop(wrapped_env)

    # Create the Pygame window with the actual frame size
    pygame_window = PygameWindow(game_loop, player_id=0, frame_size=frame_size)

    input_stream = LLMActionStream(use_buffer=True)

    # Create the LLM action wrapper
    llm_wrapper = LLMActionWrapper(game_loop, player_id=0, input_stream=input_stream)

    # Initialize Pygame and start observing the game state
    pygame_window.init_pygame()
    game_loop.setup_player_input(0, input_stream)
    await asyncio.gather(
        game_loop.game_loop(),
        pygame_window.observe_game_state(),
        llm_wrapper.consume_observations(),
    )


if __name__ == "__main__":
    asyncio.run(main())
