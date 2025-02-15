import asyncio
import fractions
import json

import ale_py
import av
import cv2
import gymnasium as gym
from aiortc import RTCDataChannel, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaStreamTrack
from fastapi import FastAPI
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware

gym.register_envs(ale_py)

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket clients connected to the backend
clients = []

# Dictionary to store current action for each channel
channel_actions = {}


class GymEnvironmentVideoTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, env_name="BreakoutNoFrameskip-v4"):
        super().__init__()
        self.env = gym.make(
            env_name,
            render_mode="rgb_array",
            full_action_space=True,
        )
        self.env.reset()
        self.frame_count = 0
        self.done = False

    async def recv(self):
        self.frame_count += 1
        frame = self.env.render()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        video_frame = av.VideoFrame.from_ndarray(frame, format="bgr24")
        video_frame.pts = self.frame_count
        video_frame.time_base = fractions.Fraction(1, 30)

        return video_frame

    def step(self, action):
        if action not in range(self.env.action_space.n):
            return None, None, None, None, {"warning": "Action out of bounds"}
        observation, reward, done, truncated, info = self.env.step(action)
        self.done = done
        if done:
            self.env.reset()
        return observation, reward, done, truncated, info

    def reset(self):
        self.env.reset()


# Define a WebSocket model for receiving and sending data
class WebRTCOffer(BaseModel):
    sdp: str
    type: str
    env_name: str  # Add environment name parameter


@app.post("/offer")
async def handle_offer(offer: WebRTCOffer):
    offer_sdp = RTCSessionDescription(sdp=offer.sdp, type=offer.type)

    pc = RTCPeerConnection()
    gym_video_track = GymEnvironmentVideoTrack(
        env_name=offer.env_name
    )  # Use the provided environment name
    pc.addTrack(gym_video_track)

    # Set up data channel for receiving messages
    @pc.on("datachannel")
    def on_data_channel(channel: RTCDataChannel):
        print(channel.label)
        print("Data channel created")

        # Initialize current action for this channel
        channel_actions[channel] = 0

        @channel.on("message")
        def on_message(message):
            handle_data_channel_message(message, channel, gym_video_track)

        # Start the game loop for this connection
        asyncio.create_task(game_loop(gym_video_track, channel))

    await pc.setRemoteDescription(offer_sdp)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}


def handle_data_channel_message(message, channel, gym_video_track):
    try:
        action = message
        if action == "reset":
            gym_video_track.reset()
            print("Environment reset")

            initial_state = gym_video_track.env.reset()
            game_state = {
                "reward": 0,
                "done": False,
            }

            # Send game state to client(s) via data channel
            channel.send(json.dumps(game_state))  # Send back to the same channel

        elif action == "start":
            channel_actions[channel] = 1  # Start action
        else:
            try:
                action = int(action)
                channel_actions[channel] = action
            except ValueError:
                print("Invalid action received:", action)

    except Exception as e:
        print("Error handling data channel message:", e)
        import traceback

        traceback.print_exc()


async def game_loop(gym_video_track, channel: RTCDataChannel):
    """
    Runs the game loop independently and applies the current action.
    """
    print(
        "Adding new game loop. Currently running:",
        len(asyncio.all_tasks()),
        "tasks",
        len(channel_actions),
        "actions",
    )
    while True:
        if channel in channel_actions:
            current_action = channel_actions[channel]
            if current_action is not None:
                state, reward, done, truncated, info = gym_video_track.step(
                    current_action
                )
                channel_actions[channel] = 0  # Reset action after applying it

                game_state = {
                    "reward": reward,
                    "done": done,
                }

                if "warning" in info:
                    game_state["warning"] = info["warning"]

                # Ensure channel is not None before sending
                if channel and channel.readyState == "open":
                    channel.send(json.dumps(game_state))

                if done:
                    gym_video_track.reset()  # Reset the game when done

        await asyncio.sleep(0.05)  # Wait for the next frame (approx 20 FPS)
        # check if the channel is closed
        if channel.readyState == "closed":
            channel_actions.pop(channel, None)
            break


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
