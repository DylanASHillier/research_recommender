import asyncio

import gymnasium as gym
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

from backend.multiplayer_games.gymnasium_compat import GymnasiumWrapper
from backend.multiplayer_games.interface import (
    AsyncAECGameLoop,
    AsyncLLMWrapper,
    BufferedInputActionStream,
    PygameWindow,
)

# Define the environment name
ENV_NAME = "LunarLander-v3"

# Define the key to action mapping
KEY_TO_ACTION = {
    "NOOP": 0,
    "left": 1,
    "main engine": 2,
    "right": 3,
}

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
device = "mps" if torch.mps.is_available() else device
# Load SmolVLM model and processor
processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-Instruct")
model = AutoModelForVision2Seq.from_pretrained(
    "HuggingFaceTB/SmolVLM-Instruct",
    torch_dtype=torch.bfloat16,
    _attn_implementation="flash_attention_2" if device == "cuda" else "eager",
).to(device)


class CustomGymnasiumWrapper(GymnasiumWrapper):
    """Custom wrapper for LunarLander-v3 environment"""

    def __init__(self, env):
        """Initialize the wrapper"""
        super().__init__(env)

    def map_action(self, action: set[str]):
        for key in action:
            if key in KEY_TO_ACTION:
                return KEY_TO_ACTION[key]
        return 0  # Default to NOOP if key not found

    @property
    def window_size(self):
        """Need to override for box2d environments"""
        return 600, 400


LLMAction = str


class LLMActionStream(BufferedInputActionStream[LLMAction]):
    def __init__(self, use_buffer=False, buffer_size=100):
        super().__init__(use_buffer=use_buffer, buffer_size=buffer_size)

    def update_state(self, event: LLMAction):
        self.current_state = event


def sub_sample_observations(observations, sample_freq=10, max_samples=2):
    """Take the last obsservation, and every nth observation
    before that"""
    if len(observations) <= sample_freq:
        return [observations[-1]][-max_samples:]
    return (
        observations[:-1][:: len(observations) - 1 // sample_freq] + [observations[-1]]
    )[-max_samples:]


class LLMActionWrapper(AsyncLLMWrapper):
    """Wrapper for LunarLander-v3 environment using LLM"""

    def __init__(self, game_loop, player_id, input_stream):
        super().__init__(game_loop, player_id, input_stream)

    async def compute_actions(self, observations) -> LLMAction:
        """Compute actions using LLM"""
        print(f"Computing actions over {len(observations)} observations")
        observations = sub_sample_observations(observations)
        images = [Image.fromarray(observation) for observation in observations]

        messages = [
            {
                "role": "user",
                "content": [
                    *[{"type": "image"} for _ in range(len(images))],
                    {
                        "type": "text",
                        "text": "What action should the agent take? Reply with 'left', 'right', or 'main engine'.",
                    },
                ],
            }
        ]

        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=prompt, images=images, return_tensors="pt").to(device)

        loop = asyncio.get_event_loop()
        generated_ids = await loop.run_in_executor(
            None, lambda: model.generate(**inputs, max_new_tokens=4)
        )
        action_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[
            0
        ].lower()

        print(action_text)
        if "left" in action_text:
            return "left"
        elif "right" in action_text:
            return "right"
        elif "main" in action_text or "engine" in action_text:
            return "main engine"
        else:
            return "NOOP"


async def main():
    env = gym.make(ENV_NAME, render_mode="rgb_array")
    env.reset()
    wrapped_env = CustomGymnasiumWrapper(env)
    frame_size = wrapped_env.window_size

    game_loop = AsyncAECGameLoop(wrapped_env)
    pygame_window = PygameWindow(game_loop, player_id=0, frame_size=frame_size)
    input_stream = LLMActionStream(use_buffer=True)
    llm_wrapper = LLMActionWrapper(game_loop, player_id=0, input_stream=input_stream)

    pygame_window.init_pygame()
    game_loop.setup_player_input(0, input_stream)
    await asyncio.gather(
        game_loop.game_loop(),
        pygame_window.observe_game_state(),
        llm_wrapper.consume_observations(),
    )


if __name__ == "__main__":
    asyncio.run(main())
