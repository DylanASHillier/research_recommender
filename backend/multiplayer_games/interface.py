"""Interface for multiplayer games

We just use a minimal version of the pettingzoo interface..."""

import abc
import asyncio
import typing

import pygame

from backend.multiplayer_games import utils

ValidActions = typing.TypeVar("ValidActions")
Observations = typing.TypeVar("Observations")
PlayerId = int | str


class MultiPlayerKeyboardMouseGameInterfaceAEC(
    abc.ABC, typing.Generic[ValidActions, Observations]
):
    """Interface for multiplayer games that use keyboard and mouse input
    Uses AEC Agent Environment Cycling"""

    @property
    def current_player(self) -> PlayerId:
        """Return the current player"""

    @property
    def window_size(self) -> tuple[int, int]:
        """Return the window size"""

    @abc.abstractmethod
    def valid_actions(self) -> ValidActions:
        """Return the valid actions for the game"""

    @abc.abstractmethod
    def reset(self, seed: int) -> None:
        """Reset Game State"""

    @abc.abstractmethod
    def step(self, actions: ValidActions, player_id: PlayerId) -> bool:
        """Take a step in the game
        Returns True if the game is over"""

    def observe(self, player_id: PlayerId) -> Observations:
        """Observe the game state"""

    def close(self) -> None:
        """Close the game"""

    def get_agents(self) -> list[PlayerId]:
        """Return the agents in the game"""


OutputStateStream = typing.AsyncIterator[Observations]


ActionEvent = typing.TypeVar("ActionEvent")


class InputActionStream(typing.Generic[ActionEvent], abc.ABC):
    """Stream of input events that can be used to control a game.
    This just needs to output actions asynchronously."""

    def __init__(self, sync: bool = False):
        self.sync = sync
        self.sync_event = asyncio.Event() if sync else None

    async def get_sync_state(self):
        """Wait for the next sync event"""
        assert self.sync
        await self.sync_event.wait()
        self.sync_event.clear()
        await self.get_state()

    def set_sync_state(self):
        """Set the sync event"""
        assert self.sync
        self.sync_event.set()

    @abc.abstractmethod
    async def get_state(self) -> ActionEvent:
        """Returns the current keyboard state asynchronously."""
        raise NotImplementedError


class BufferedInputActionStream(InputActionStream[ActionEvent]):
    """Stream of input events that can be used to control a game."""

    def __init__(self, use_buffer=True, buffer_size=100, sync=False):
        """
        :param use_buffer: If True, stream from buffer instead of current state.
        :param buffer_size: Maximum size of the event buffer.

        Generally this needs to be subscribed to a keyboard event source
        """
        self.use_buffer = use_buffer
        self.buffer = asyncio.Queue(maxsize=buffer_size) if use_buffer else None
        self.current_state = None
        self.lock = asyncio.Lock()
        super().__init__(sync=sync)

    @abc.abstractmethod
    def update_state(self, event: ActionEvent):
        """Handles an input event and updates state or buffer."""
        raise NotImplementedError

    async def handle_event(self, event: ActionEvent):
        """Handles an input event and updates state or buffer asynchronously."""
        async with self.lock:
            self.update_state(event)

            if self.use_buffer:
                if self.buffer.full():
                    await self.buffer.get()  # Prevent blocking by discarding oldest event
                await self.buffer.put(
                    self.current_state
                )  # Store a snapshot of the state

    async def get_state(self):
        """Returns the current keyboard state or an event from the buffer asynchronously."""
        async with self.lock:
            if self.use_buffer and not self.buffer.empty():
                return await self.buffer.get()
            return self.current_state  # Return a copy to avoid race conditions.


class AECGameLoop(abc.ABC, typing.Generic[ValidActions, Observations, ActionEvent]):
    """General Game Loop for AEC Games"""

    def __init__(
        self, env: MultiPlayerKeyboardMouseGameInterfaceAEC[ValidActions, Observations]
    ):
        self.env = env
        self.num_players = len(env.get_agents())
        self.channel_actions: dict[PlayerId, InputActionStream[ActionEvent] | None] = {
            player_id: None for player_id in self.env.get_agents()
        }

    def setup_player_input(
        self, player_id: PlayerId, action_stream: InputActionStream[ActionEvent]
    ):
        """Setup the input for a player"""
        self.channel_actions[player_id] = action_stream

    @abc.abstractmethod
    async def game_loop(self):
        """Runs the game loop independently and applies the current action."""
        raise NotImplementedError

    @abc.abstractmethod
    async def emit_observations(self, player_id: PlayerId) -> OutputStateStream:
        """Returns an iterator that yields the observations of the game for a player."""
        raise NotImplementedError


class SynchronousAECGameLoop(AECGameLoop[ValidActions, Observations, ActionEvent]):
    """A synchronous game loop, designed for turn-based games.
    Runs the game loop in the background and sends the observations to subscribers, whenever
    the game state changes. Waits for the action to be applied before continuing the loop.
    """

    def __init__(
        self,
        env: MultiPlayerKeyboardMouseGameInterfaceAEC[ValidActions, Observations],
    ):
        self.env = env
        self.num_players = len(env.get_agents())
        self.channel_actions: dict[PlayerId, InputActionStream[ActionEvent] | None] = {
            player_id: None for player_id in self.env.get_agents()
        }
        self.state_changed: dict[PlayerId, asyncio.Event] = {
            player_id: asyncio.Event() for player_id in self.env.get_agents()
        }
        self.change_observed: dict[PlayerId, asyncio.Event] = {
            player_id: asyncio.Event() for player_id in self.env.get_agents()
        }

    def setup_player_input(
        self, player_id: PlayerId, action_stream: InputActionStream[ActionEvent]
    ):
        """Setup the input for a player"""
        self.channel_actions[player_id] = action_stream

    async def game_loop(self):
        """Runs the game loop independently and applies the current action."""
        self.env.reset()
        print(
            "Adding new game loop. Currently running:",
            len(asyncio.all_tasks()),
            "tasks",
            len(self.channel_actions),
            "actions",
        )
        print(self.channel_actions)
        assert all(
            action_stream is not None for action_stream in self.channel_actions.values()
        )
        assert all(
            action_stream.sync for action_stream in self.channel_actions.values()
        )
        while True:
            for player_id in self.env.get_agents():
                action = await self.channel_actions[player_id].get_state()
                if action is not None:
                    done = self.env.step(action, player_id)
                    if done:
                        self.env.reset()
                    for player_id in self.env.get_agents():
                        self.state_changed.set()
                    await asyncio.gather(
                        *[
                            self.change_observed[player_id].wait()
                            for player_id in self.env.get_agents()
                        ]
                    )
                    for player_id in self.env.get_agents():
                        self.change_observed[player_id].clear()

    async def emit_observations(self, player_id: PlayerId) -> OutputStateStream:
        """Returns an iterator that yields the observations of the game for a player."""
        while True:
            await self.state_changed[player_id].wait()
            self.state_changed[player_id].clear()
            yield self.env.observe(player_id)
            self.change_observed[player_id].set()


class AsyncAECGameLoop(AECGameLoop[ValidActions, Observations, ActionEvent]):
    """An async game loop. Tries to run the game in the background at a fixed simulation rate
    and sends the observations to subscribers.

    Keeps seperate input states for each player that can be updated by the server.
    Methods:
    - game_loop: Runs the game loop independently and applies the current action.
    - get_observations: Returns an iterator that yields the observations of the game for a player.
    """

    def __init__(
        self,
        env: MultiPlayerKeyboardMouseGameInterfaceAEC[ValidActions, Observations],
        simulation_rate: float = 60,
        fps: int = 60,
    ):
        super().__init__(
            env,
        )
        self.simulation_rate = simulation_rate
        self.fps = fps

    async def game_loop(self):
        """Runs the game loop independently and applies the current action."""
        self.env.reset()
        print(
            "Adding new game loop. Currently running:",
            len(asyncio.all_tasks()),
            "tasks",
            len(self.channel_actions),
            "actions",
        )
        print(self.channel_actions)
        assert all(
            action_stream is not None for action_stream in self.channel_actions.values()
        )
        rate_limiter = utils.AsyncRateLimiter(self.simulation_rate)
        while True:
            async with rate_limiter:
                for player_id in self.env.get_agents():
                    action = await self.channel_actions[player_id].get_state()
                    if action is not None:
                        done = self.env.step(action, player_id)
                        if done:
                            self.env.reset()

    async def emit_observations(self, player_id: PlayerId) -> OutputStateStream:
        """Returns an iterator that yields the observations of the game for a player."""
        rate_limiter = utils.AsyncRateLimiter(self.fps)
        while True:
            async with rate_limiter:
                yield self.env.observe(player_id)


RGBArray = typing.Any


class PygameWindow:
    """A subscriber that displays the game state in a Pygame window.
    Also allows the player to interact with the game using keyboard and mouse input."""

    def __init__(
        self,
        game_loop: AECGameLoop[ValidActions, RGBArray, ActionEvent],
        player_id: PlayerId,
        frame_size: tuple[int, int],
    ):
        self.game_loop = game_loop
        self.player_id = player_id
        self.frame_size = frame_size
        self.screen = ...
        self.clock = ...

    # pylint: disable=no-member
    def init_pygame(self):
        """Initialize Pygame"""
        pygame.init()
        self.screen = pygame.display.set_mode(self.frame_size)
        pygame.display.set_caption("TextArena Game")
        self.clock = pygame.time.Clock()

    # pylint: enable=no-member

    async def observe_game_state(self):
        """Observe the game state and display it in a Pygame window"""
        async for observation in self.game_loop.emit_observations(self.player_id):
            frame = pygame.surfarray.make_surface(observation.swapaxes(0, 1))
            self.screen.blit(frame, (0, 0))
            pygame.display.flip()
            self.clock.tick(60)


class FrameSkipWrapper(OutputStateStream):
    """Wrapper for AEC Game Loop OutputStateStream. Skips frames from the game loop
    When the game is run, it outputs observations continuously. For slow subscribers,
    this wrapper can be used to skip frames and only return every nth frame.
    """

    def __init__(self, original_output_stream: OutputStateStream, skip_frames: int = 1):
        self.output_stream = original_output_stream
        self.skip_frames = skip_frames
        self.frame_count = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        observations = await self.output_stream.__anext__()
        self.frame_count += 1
        if self.frame_count % self.skip_frames == 0:
            return observations
        else:
            return await self.__anext__()


class ObservationAccumulationWrapper(OutputStateStream):
    """Wrapper for AEC Game Loop OutputStateStream. Accumulates observations from the game loop
    When the game is run, it outputs observations continuously, which may be too
    fast for the subscriber to consume. This wrapper accumulates the observations
    accumulated since the last call to get_observations and returns them all at once.

    In essence this uses one loop to accumulate observations and another loop to consume them
    instantenously whenever the subscriber is ready to consume them.
    """

    def __init__(self, original_output_stream: OutputStateStream):
        self.output_stream = original_output_stream
        self.accumulated_observations = []
        self.lock = asyncio.Lock()
        asyncio.create_task(self.accumulate_observations())

    async def accumulate_observations(self):
        """Accumulate observations from the original output stream."""
        async for observation in self.output_stream:
            async with self.lock:
                self.accumulated_observations.append(observation)

    async def get_observations(self) -> typing.List[Observations]:
        """Return the accumulated observations and clear the buffer."""
        async with self.lock:
            observations = self.accumulated_observations.copy()
            self.accumulated_observations.clear()
            return observations

    def __aiter__(self):
        return self

    async def __anext__(self):
        observations = await self.get_observations()
        if observations:
            return observations
        else:
            await asyncio.sleep(0.15)  # Prevent busy waiting
            return await self.__anext__()


class AsyncLLMWrapper(abc.ABC):
    """Wrapper for LLM Subscriber to AsyncAECGameLoop

    The wrapper should emit an InputActionStream and consume an OutputStateStream"""

    def __init__(
        self,
        game_loop: AsyncAECGameLoop[ValidActions, Observations, ActionEvent],
        player_id: PlayerId,
        input_stream: BufferedInputActionStream,
    ):
        self.game_loop = game_loop
        self.player_id = player_id
        self.input_stream = input_stream

    async def emit_actions(self, actions: ValidActions):
        """Emit actions to the game loop"""
        await self.input_stream.handle_event(actions)
        if self.input_stream.sync:
            self.input_stream.set_sync_state()

    @abc.abstractmethod
    async def compute_actions(self, observation: Observations) -> ValidActions:
        """Compute the actions for a player"""

    async def consume_observations(self, observation_stream: OutputStateStream):
        """Consume observations from the game loop"""
        async for observations in observation_stream:
            actions = await self.compute_actions(observations)
            await self.emit_actions(actions)
