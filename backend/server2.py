import json
import asyncio
import gymnasium as gym
import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCDataChannel
from starlette.middleware.cors import CORSMiddleware
from aiortc.contrib.media import MediaStreamTrack
import fractions
import av
import ale_py

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

# Gym environment setup (Breakout)
gym_video_track = None

# WebSocket clients connected to the backend
clients = []

# Global variable for action
current_action = 0


class GymEnvironmentVideoTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, env_name="BreakoutNoFrameskip-v4"):
        super().__init__()
        self.env = gym.make(env_name, render_mode="rgb_array")
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
        observation, reward, done, truncated, info = self.env.step(action)
        self.done = done
        if done:
            self.env.reset()
        return observation, reward, done, truncated, info

    def reset(self):
        self.env.reset()


gym_video_track = GymEnvironmentVideoTrack()


# Define a WebSocket model for receiving and sending data
class WebRTCOffer(BaseModel):
    sdp: str
    type: str


@app.post("/offer")
async def handle_offer(offer: WebRTCOffer):
    offer_sdp = RTCSessionDescription(sdp=offer.sdp, type=offer.type)

    pc = RTCPeerConnection()
    pc.addTrack(gym_video_track)

    # Set up data channel for receiving messages
    @pc.on("datachannel")
    def on_data_channel(channel: RTCDataChannel):
        print(channel.label)
        print("Data channel created")

        @channel.on("message")
        def on_message(message):
            # print(f"Received from client: {message}")
            handle_data_channel_message(message, channel)

    await pc.setRemoteDescription(offer_sdp)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}


def handle_data_channel_message(message, channel):
    try:
        action = message
        global current_action
        if action == "reset":
            gym_video_track.reset()
            print("Environment reset")

            initial_state = gym_video_track.env.reset()
            game_state = {
                "reward": 0,
                "done": False,
                # "state": initial_state[0].tolist(),  # Send the initial state
            }

            # Send game state to client(s) via data channel
            channel.send(json.dumps(game_state))  # Send back to the same channel

        elif action == "start":

            current_action = 1  # Start action
        elif action in ["0", "1", "2", "3"]:
            current_action = int(action)

    except Exception as e:
        print("Error handling data channel message:", e)
        import traceback

        traceback.print_exc()


async def game_loop():
    """
    Runs the game loop independently and applies the current action.
    """
    global current_action

    while True:
        if current_action is not None:
            state, reward, done, truncated, info = gym_video_track.step(current_action)
            current_action = 0  # Reset action after applying it

            game_state = {
                "reward": reward,
                "done": done,
            }

            # Send game state to WebRTC clients via data channel
            for client in clients:
                await client.send_text(json.dumps(game_state))

            if done:
                gym_video_track.reset()  # Reset the game when done

        await asyncio.sleep(0.05)  # Wait for the next frame (approx 20 FPS)


@app.on_event("startup")
async def startup():
    """
    Start the background task for the game loop
    """
    asyncio.create_task(game_loop())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
