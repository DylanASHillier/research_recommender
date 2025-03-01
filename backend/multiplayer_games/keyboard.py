import asyncio
import enum
import typing
import pygame
import dataclasses
from backend.multiplayer_games import interface


class KeyAction(str, enum.Enum):
    """Enum for key actions"""

    PRESS = "press"
    RELEASE = "release"


class ModifierKey(str, enum.Enum):
    """Enum for modifier keys"""

    SHIFT = "shift"
    CTRL = "ctrl"
    ALT = "alt"
    META = "meta"  # Command key on macOS, Windows key on Windows


class Key(str, enum.Enum):
    """All possible keys"""

    A = "a"
    B = "b"
    C = "c"
    D = "d"
    E = "e"
    F = "f"
    G = "g"
    H = "h"
    I = "i"
    J = "j"
    K = "k"
    L = "l"
    M = "m"
    N = "n"
    O = "o"
    P = "p"
    Q = "q"
    R = "r"
    S = "s"
    T = "t"
    U = "u"
    V = "v"
    W = "w"
    X = "x"
    Y = "y"
    Z = "z"
    ZERO = "0"
    ONE = "1"
    TWO = "2"
    THREE = "3"
    FOUR = "4"
    FIVE = "5"
    SIX = "6"
    SEVEN = "7"
    EIGHT = "8"
    NINE = "9"
    SPACE = "space"
    ENTER = "enter"
    BACKSPACE = "backspace"
    TAB = "tab"
    ESCAPE = "escape"
    DELETE = "delete"
    INSERT = "insert"
    PRINT_SCREEN = "printscreen"
    PAUSE = "pause"
    BREAK = "break"
    CONTEXT_MENU = "contextmenu"
    GRAVE = "grave"
    MINUS = "minus"
    EQUAL = "equal"
    BRACKET_LEFT = "bracketleft"
    BRACKET_RIGHT = "bracketright"
    BACKSLASH = "backslash"
    SEMICOLON = "semicolon"
    QUOTE = "quote"
    COMMA = "comma"
    PERIOD = "period"
    SLASH = "slash"
    ARROW_UP = "arrowup"
    ARROW_DOWN = "arrowdown"
    ARROW_LEFT = "arrowleft"
    ARROW_RIGHT = "arrowright"
    PAGE_UP = "pageup"
    PAGE_DOWN = "pagedown"
    HOME = "home"
    END = "end"
    CAPS_LOCK = "capslock"
    NUM_LOCK = "numlock"
    SCROLL_LOCK = "scrolllock"
    F1 = "f1"
    F2 = "f2"
    F3 = "f3"
    F4 = "f4"
    F5 = "f5"
    F6 = "f6"
    F7 = "f7"
    F8 = "f8"
    F9 = "f9"
    F10 = "f10"
    F11 = "f11"
    F12 = "f12"
    NUMPAD_0 = "numpad0"
    NUMPAD_1 = "numpad1"
    NUMPAD_2 = "numpad2"
    NUMPAD_3 = "numpad3"
    NUMPAD_4 = "numpad4"
    NUMPAD_5 = "numpad5"
    NUMPAD_6 = "numpad6"
    NUMPAD_7 = "numpad7"
    NUMPAD_8 = "numpad8"
    NUMPAD_9 = "numpad9"
    NUMPAD_DECIMAL = "numpaddecimal"
    NUMPAD_DIVIDE = "numpaddivide"
    NUMPAD_MULTIPLY = "numpadmultiply"
    NUMPAD_SUBTRACT = "numpadsubtract"
    NUMPAD_ADD = "numpadadd"
    NUMPAD_ENTER = "numpadenter"


KeyboardActionEvent = typing.TypeVar(
    "KeyboardActionEvent", bound=typing.Dict[str, typing.Union[str, int]]
)


@dataclasses.dataclass
class MousePos:
    """Enum for mouse positions"""

    x: int
    y: int


@dataclasses.dataclass
class AllInputs:
    """All inputs for a player"""

    keyboard: set[Key]
    mouse_pos: MousePos | None


HasMousePos = bool


class KeyboardActionStream(interface.BufferedInputActionStream[KeyboardActionEvent]):
    """Stream of keyboard input events."""

    def __init__(self, use_buffer=False, buffer_size=100):
        super().__init__(use_buffer=use_buffer, buffer_size=buffer_size)
        self.current_state = set()

    def update_state(self, event: KeyboardActionEvent):
        if event["type"] == "keydown":
            self.current_state.add(event["key"])
        elif event["type"] == "keyup":
            self.current_state.discard(event["key"])


# pylint: disable=no-member
class PygameKeyboardWrapper(KeyboardActionStream):
    """Wrapper around KeyboardStateStream for Pygame input handling."""

    def __init__(self, use_buffer=False, buffer_size=100):
        """
        Wrapper around KeyboardStateStream for Pygame input handling.

        :param use_buffer: If True, use a buffer instead of real-time state.
        :param buffer_size: Max size of event buffer.
        """
        super().__init__(use_buffer, buffer_size)
        self.running = False
        # start the event processing loop
        asyncio.create_task(self.process_pygame_events())

    async def process_pygame_events(self):
        """Continuously listens for Pygame keyboard events and updates the state."""
        self.running = True
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False  # Stop loop if quit event occurs
                elif event.type in (pygame.KEYDOWN, pygame.KEYUP):
                    await self.handle_event(
                        {
                            "type": (
                                "keydown" if event.type == pygame.KEYDOWN else "keyup"
                            ),
                            "key": pygame.key.name(
                                event.key
                            ),  # Converts keycode to string
                        }
                    )
            await asyncio.sleep(0.01)  # Prevents high CPU usage

    def stop(self):
        """Stops event processing."""
        self.running = False


# pylint: enable=no-member
