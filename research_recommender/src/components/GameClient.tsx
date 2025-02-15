import React, { useEffect, useRef, useState } from 'react';

interface GameState {
    reward: number;
    done: boolean;
    warning?: string;
}

const atariEnvironments = [
    "BreakoutNoFrameskip-v4",
    "PongNoFrameskip-v4",
    "SpaceInvadersNoFrameskip-v4",
    "MsPacmanNoFrameskip-v4",
    "SeaquestNoFrameskip-v4"
];

const atariActions = [
    { key: "0", description: "NOOP", keyboard: "N" },
    { key: "1", description: "FIRE", keyboard: "F" },
    { key: "2", description: "UP", keyboard: "W" },
    { key: "3", description: "RIGHT", keyboard: "D" },
    { key: "4", description: "LEFT", keyboard: "A" },
    { key: "5", description: "DOWN", keyboard: "S" },
    { key: "6", description: "UPRIGHT", keyboard: "E" },
    { key: "7", description: "UPLEFT", keyboard: "Q" },
    { key: "8", description: "DOWNRIGHT", keyboard: "C" },
    { key: "9", description: "DOWNLEFT", keyboard: "Z" },
    { key: "10", description: "UPFIRE", keyboard: "R" },
    { key: "11", description: "RIGHTFIRE", keyboard: "T" },
    { key: "12", description: "LEFTFIRE", keyboard: "G" },
    { key: "13", description: "DOWNFIRE", keyboard: "B" },
    { key: "14", description: "UPRIGHTFIRE", keyboard: "Y" },
    { key: "15", description: "UPLEFTFIRE", keyboard: "U" },
    { key: "16", description: "DOWNRIGHTFIRE", keyboard: "I" },
    { key: "17", description: "DOWNLEFTFIRE", keyboard: "O" },
    { key: "40", description: "RESET1", keyboard: "P" }
];

const GameClient = () => {
    const videoRef = useRef<HTMLVideoElement>(null);
    const peerConnectionRef = useRef<RTCPeerConnection | null>(null);
    const dataChannelRef = useRef<RTCDataChannel | null>(null); // For sending game actions
    const [gameState, setGameState] = useState<GameState | null>(null);
    const [isConnected, setIsConnected] = useState(false);
    const [videoStatus, setVideoStatus] = useState<string>("No video");
    const [selectedEnv, setSelectedEnv] = useState<string>(atariEnvironments[0]);

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

                if (!process.env.NEXT_PUBLIC_GAME_ENDPOINT) {
                    throw new Error("GAME_ENDPOINT is not defined");
                }
                const response = await fetch(process.env.NEXT_PUBLIC_GAME_ENDPOINT + "/offer", {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        sdp: pc.localDescription?.sdp,
                        type: pc.localDescription?.type,
                        env_name: selectedEnv  // Send the selected environment name
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
    }, [selectedEnv]);

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

            const action = atariActions.find(a => a.keyboard === event.key.toUpperCase());
            if (action) {
                console.log(`Sending action: ${action.description}`);
                sendAction(action.key);
            }
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
        <div className="max-w-2xl mx-auto p-4 text-white">
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
                <div className="mt-4 p-4 bg-gray-800 rounded-lg">
                    <h3 className="font-bold mb-2">Game State</h3>
                    <div>
                        <p>Reward: {gameState.reward}</p>
                        <p>Done: {gameState.done ? 'Yes' : 'No'}</p>
                        {gameState.warning && <p className="text-red-500">Warning: {gameState.warning}</p>}
                    </div>
                </div>
            )}

            {/* Environment Selection */}
            <div className="mt-4">
                <label htmlFor="env-select" className="block mb-2 font-bold">Select Environment:</label>
                <select
                    id="env-select"
                    value={selectedEnv}
                    onChange={(e) => setSelectedEnv(e.target.value)}
                    className="p-2 border rounded-lg bg-gray-700 text-white"
                >
                    {atariEnvironments.map((env) => (
                        <option key={env} value={env}>{env}</option>
                    ))}
                </select>
            </div>

            {/* Action Guide */}
            <div className="mt-4 p-4 bg-gray-800 rounded-lg">
                <h3 className="font-bold mb-2">Action Guide</h3>
                <ul>
                    {atariActions.map((action) => (
                        <li key={action.key}>
                            <strong>{action.keyboard}</strong>: {action.description}
                        </li>
                    ))}
                </ul>
            </div>

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
