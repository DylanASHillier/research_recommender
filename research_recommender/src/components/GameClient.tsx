import React, { useEffect, useRef, useState } from 'react';

interface GameState {
    reward: number;
    done: boolean;
}

const GameClient = () => {
    const videoRef = useRef<HTMLVideoElement>(null);
    const peerConnectionRef = useRef<RTCPeerConnection | null>(null);
    const dataChannelRef = useRef<RTCDataChannel | null>(null); // For sending game actions
    const [gameState, setGameState] = useState<GameState | null>(null);
    const [isConnected, setIsConnected] = useState(false);
    const [videoStatus, setVideoStatus] = useState<string>("No video");

    useEffect(() => {
        const setupWebRTC = async () => {
            try {
                console.log("Setting up WebRTC connection...");
                const pc = new RTCPeerConnection({
                    iceServers: [{ urls: 'stun:stun.l.google.com:19302' }]
                });
                peerConnectionRef.current = pc;

                // Create a RTCDataChannel for sending actions
                const dataChannel = pc.createDataChannel("actions");
                dataChannelRef.current = dataChannel;

                // Handle the data channel opening and receiving actions
                dataChannel.onopen = () => {
                    console.log("Data channel opened for actions");
                };

                dataChannel.onmessage = (event) => {
                    console.log("Received action:", event.data);
                    const state = JSON.parse(event.data);
                    setGameState(state);
                };

                pc.onicecandidate = (event) => {
                    console.log("New ICE candidate:", event.candidate);
                };

                pc.oniceconnectionstatechange = () => {
                    console.log("ICE Connection State:", pc.iceConnectionState);
                    setVideoStatus(`ICE state: ${pc.iceConnectionState}`);
                };

                pc.onconnectionstatechange = () => {
                    console.log("Connection State:", pc.connectionState);
                    setVideoStatus(`Connection state: ${pc.connectionState}`);
                };

                pc.ontrack = (event) => {
                    console.log("Received track:", event.track.kind);
                    setVideoStatus("Received video track");

                    if (videoRef.current && event.streams[0]) {
                        videoRef.current.srcObject = event.streams[0];
                        console.log("Set video source");

                        // Log stream and track info
                        const stream = event.streams[0];
                        console.log("Stream info:", {
                            active: stream.active,
                            id: stream.id,
                            tracks: stream.getTracks().map(t => ({
                                kind: t.kind,
                                enabled: t.enabled,
                                muted: t.muted,
                                readyState: t.readyState
                            }))
                        });
                    }
                };

                // Create and send offer
                const offer = await pc.createOffer({
                    offerToReceiveVideo: true
                });
                await pc.setLocalDescription(offer);

                const response = await fetch('https://ec43-202-161-44-3.ngrok-free.app/offer', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        sdp: pc.localDescription?.sdp,
                        type: pc.localDescription?.type
                    })
                });

                if (!response.ok) {
                    throw new Error(`Server responded with ${response.status}`);
                }

                const answer = await response.json();
                await pc.setRemoteDescription(answer);
                setIsConnected(true);

            } catch (err: unknown) {
                if (err instanceof Error) {
                    console.error("WebRTC setup error:", err);
                    setVideoStatus(`Error: ${err.message}`);
                } else {
                    console.error("WebRTC setup error:", err);
                    setVideoStatus("An unknown error occurred");
                }
            }
        };

        setupWebRTC();

        return () => {
            peerConnectionRef.current?.close();
        };
    }, []);

    // Send actions via RTCDataChannel
    const sendAction = (action: string) => {
        if (dataChannelRef.current?.readyState === "open") {
            dataChannelRef.current.send(action); // Send the action to the server
            console.log(`Sent action: ${action}`);
        } else {
            console.error("Data channel is not open.");
        }
    };

    // Listen for keyboard input and send actions
    useEffect(() => {
        const handleKeyPress = (event: KeyboardEvent) => {
            if (!isConnected) return;

            let action = null;
            switch (event.key) {
                case 'ArrowLeft':
                    action = "3";
                    console.log("Sending action: LEFT");
                    break;
                case 'ArrowRight':
                    action = "2";
                    console.log("Sending action: RIGHT");
                    break;
                case ' ':
                    action = "1";
                    console.log("Sending action: START");
                    break;
                default:
                    return;
            }

            sendAction(action);
        };

        window.addEventListener('keydown', handleKeyPress);
        return () => window.removeEventListener('keydown', handleKeyPress);
    }, [isConnected]);

    // Handle reset button click
    const handleReset = () => {
        if (dataChannelRef.current && isConnected) {
            console.log("Sending reset signal...");
            sendAction("reset");
        }
    };

    const handleStart = () => {
        if (dataChannelRef.current && isConnected) {
            console.log("Sending start signal...");
            sendAction("start");
        }
    };

    return (
        <div className="max-w-2xl mx-auto p-4">
            <div className="relative bg-black rounded-lg overflow-hidden" style={{ aspectRatio: '16/9' }}>
                <video
                    ref={videoRef}
                    autoPlay
                    playsInline
                    className="w-full h-full object-contain"
                    onLoadedMetadata={() => {
                        console.log("Video metadata loaded");
                        setVideoStatus("Metadata loaded");
                    }}
                    onPlay={() => {
                        console.log("Video started playing");
                        setVideoStatus("Playing");
                    }}
                    onError={(e) => {
                        console.error("Video error:", e);
                        setVideoStatus(`Video error: ${e}`);
                    }}
                />
                <div className="absolute top-0 left-0 p-2 text-white bg-black bg-opacity-50">
                    {videoStatus}
                </div>
            </div>

            {gameState && (
                <div className="mt-4 p-4 bg-gray-100 rounded-lg">
                    <h3 className="font-bold mb-2">Game State</h3>
                    <div>
                        <p>Reward: {gameState.reward}</p>
                        <p>Done: {gameState.done ? 'Yes' : 'No'}</p>
                    </div>
                </div>
            )}

            {/* Reset Button */}
            <button
                onClick={handleReset}
                className="mt-4 p-2 bg-blue-500 text-white rounded-lg"
            >
                Reset Game
            </button>
            <button
                onClick={handleStart}
                className="mt-4 p-2 bg-blue-500 text-white rounded-lg"
            >
                Start Game
            </button>
        </div>
    );
};

export default GameClient;
