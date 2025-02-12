import json
import asyncio
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from aiortc import RTCPeerConnection, RTCSessionDescription

app = FastAPI()

# Store active WebRTC connections
peers = {}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket Connected")

    peer_connection = RTCPeerConnection()
    peers[websocket] = peer_connection

    @peer_connection.on("datachannel")
    def on_datachannel(channel):
        @channel.on("message")
        async def on_message(message):
            print(f"Received from client: {message}")
            await asyncio.sleep(0.05)  # Simulating processing time
            channel.send(f"Echo: {message}")  # Dummy AI response

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            if message["type"] == "offer":
                print("Received WebRTC Offer")
                offer = RTCSessionDescription(sdp=message["sdp"], type="offer")
                await peer_connection.setRemoteDescription(offer)

                answer = await peer_connection.createAnswer()
                await peer_connection.setLocalDescription(answer)

                await websocket.send_text(
                    json.dumps(
                        {"type": "answer", "sdp": peer_connection.localDescription.sdp}
                    )
                )
                print("Sent WebRTC Answer")

    except WebSocketDisconnect:
        print("WebSocket Disconnected")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await peer_connection.close()
        peers.pop(websocket, None)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
