import gymnasium as gym

# Create your game environment
env = gym.make("Pong-v4")


def process_input(input_data):
    if input_data["type"] == "key":
        key = input_data["key"]
        action = map_key_to_action(key)  # Implement this mapping
        env.step(action)  # Send action to Gym

    elif input_data["type"] == "mouse":
        print(f"Mouse at ({input_data['x']}, {input_data['y']})")  # Example use


def map_key_to_action(key):
    key_map = {"ArrowUp": 2, "ArrowDown": 3}  # Example Pong actions
    return key_map.get(key, 0)  # Default to "no-op"
