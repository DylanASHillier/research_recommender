"""Example demonstrating how to locally run a PettingZoo game with a human player and an LLM player"""

import asyncio
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

from pettingzoo.atari import surround_v2
from backend.multiplayer_games.pettingzoo_compat import PettingZooWrapper
from backend.multiplayer_games.interface import (
    AsyncAECGameLoop,
    AsyncLLMWrapper,
    InputActionStream,
    BufferedInputActionStream,
    PygameWindow,
    ObservationAccumulationWrapper,
    FrameSkipWrapper,
    OutputStateStream,
)
from backend.multiplayer_games.keyboard import PygameKeyboardWrapper

# Define the key to action mapping
KEY_TO_ACTION = {
    "space": 1,
    "up": 2,
    "left": 4,
    "right": 3,
    "down": 5,
    "noop": 0,
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


class CustomPettingZooWrapper(PettingZooWrapper):
    """Custom wrapper for PettingZoo environment"""

    def map_action(self, action: int):
        return action


LLMAction = int


class LLMActionStream(BufferedInputActionStream[LLMAction]):
    def __init__(self, use_buffer=False, buffer_size=100):
        super().__init__(use_buffer=use_buffer, buffer_size=buffer_size)
        self.current_state = 0  # Default to NOOP

    def update_state(self, event: LLMAction):
        self.current_state = event


def sub_sample_observations(observations, sample_freq=10, max_samples=2):
    """Take the last obsservation, and every sample_freq'th observation
    before that"""
    if len(observations) <= sample_freq:
        return [observations[-1]][-max_samples:]
    return (
        observations[:-1][:: len(observations) - 1 // sample_freq] + [observations[-1]]
    )[-max_samples:]


class LLMActionWrapper(AsyncLLMWrapper):
    """Wrapper for PettingZoo environment using LLM"""

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
                        "text": "What action should the agent take? Reply with {ACTIONS}.".format(
                            ACTIONS="fire,left,right,up,down,noop"
                        ),
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
        for key, action in KEY_TO_ACTION.items():
            if key in action_text:
                return action
        return 0


class SingleActionWrapper(InputActionStream[LLMAction]):
    def __init__(self, input_stream: PygameKeyboardWrapper):
        self.input_stream = input_stream

    async def get_state(self):
        action_set: set = await self.input_stream.get_state()
        for key in action_set:
            if key in KEY_TO_ACTION:
                return KEY_TO_ACTION[key]
        return 0


def llm_observation_stream_wrapper(observation_stream: OutputStateStream):
    return ObservationAccumulationWrapper(FrameSkipWrapper(observation_stream, 10))


async def main():
    env = surround_v2.env(full_action_space=True)
    env.reset()
    wrapped_env = CustomPettingZooWrapper(env)
    frame_size = wrapped_env.window_size

    game_loop = AsyncAECGameLoop(wrapped_env, simulation_rate=15, fps=30)
    pygame_window = PygameWindow(game_loop, player_id="first_0", frame_size=frame_size)
    input_stream = PygameKeyboardWrapper()
    input_stream = SingleActionWrapper(input_stream)
    llm_input_stream = LLMActionStream(use_buffer=True)
    llm_wrapper = LLMActionWrapper(
        game_loop, player_id="second_0", input_stream=llm_input_stream
    )

    pygame_window.init_pygame()
    game_loop.setup_player_input("first_0", input_stream)
    game_loop.setup_player_input("second_0", llm_input_stream)
    await asyncio.gather(
        game_loop.game_loop(),
        pygame_window.observe_game_state(),
        llm_wrapper.consume_observations(
            llm_observation_stream_wrapper(game_loop.emit_observations("second_0"))
        ),
    )


if __name__ == "__main__":
    asyncio.run(main())
