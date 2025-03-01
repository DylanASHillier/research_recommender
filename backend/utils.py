import os, uuid, asyncio, json, time, math
from tabulate import tabulate
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import logging
import logging.config
from logging.handlers import RotatingFileHandler
from starlette.websockets import WebSocketState
from supabase import create_client, Client

import textarena as ta

from config import (
    SUPABASE_URL,
    SUPABASE_KEY,
    HUMANITY_MODEL_ID,
    HUMAN_K_FACTOR,
    STANDARD_MODEL_IDS,
    STANDARD_MODEL_K_FACTOR,
    GAMES_THRESHOLD,
    INITIAL_K,
    REDUCED_K,
    DEFAULT_ELO,
    ACCEPTABLE_TIME,
    MAX_ELO_DELTA,
    BACKGROUND_LOOP_INTERVAL,
    TIMEOUT_LIMIT,
    DOUBLE_STANDARD_PENALTY,
    NUM_ENVIRONMENTS_PER_BATCH,
    SNAPSHOT_INTERVAL,
)


# --------------------
# Logging Configuration
# --------------------
class DebugOnlyFilter(logging.Filter):
    def filter(self, record):
        return record.levelno == logging.DEBUG


LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "detailed": {"format": "%(asctime)s [%(levelname)s] %(name)s - %(message)s"},
        "simple": {"format": "%(levelname)s: %(message)s"},
    },
    "filters": {"debug_only": {"()": DebugOnlyFilter}},
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "simple",
            "level": "INFO",
            "stream": "ext://sys.stdout",
        },
        "full_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "detailed",
            "filename": "full.log",
            "maxBytes": 10 * 1024 * 1024,  # 10 MB
            "backupCount": 5,
            "level": "DEBUG",
        },
        "debug_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "detailed",
            "filename": "debug.log",
            "maxBytes": 10 * 1024 * 1024,
            "backupCount": 5,
            "level": "DEBUG",
            "filters": ["debug_only"],
        },
        "info_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "detailed",
            "filename": "info.log",
            "maxBytes": 10 * 1024 * 1024,
            "backupCount": 5,
            "level": "INFO",
        },
    },
    "root": {
        "handlers": ["console", "full_file", "debug_file", "info_file"],
        "level": "DEBUG",
    },
}
logging.config.dictConfig(LOGGING_CONFIG)


# --------------------
# Supabase Client
# --------------------
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


# --------------------------------
# Global Data Structures & Helpers
# --------------------------------
active_connections = {}  # { token: {...connection details...} }
active_matches = {}  # { match_token: {...match details...} }
general_information = {"past_queue_times": []}


def load_matchmaking_registry():
    """
    Retrieve all active environments from Supabase and build a registry
    structure, e.g.:
       {
         env_id: {
           "num_players": int,
           "env_name": str,
           "queue": {}   # tokens queued
         },
         ...
       }
    """
    environments = (
        supabase.table("environments")
        .select("id, num_players, env_name, active")
        .eq("active", True)
        .execute()
    )
    return_dict = {}
    for env_dict in environments.data:
        return_dict[env_dict["id"]] = {
            "num_players": env_dict["num_players"],
            "env_name": env_dict["env_name"],
            "queue": {},
        }
    return return_dict


matchmaking_registry = load_matchmaking_registry()

matchmaking_batch_index = 0


# --------------------
# ELO Calculation
# --------------------
def get_dynamic_k(model_id: int, games_played: int) -> float:
    """Return K-factor for the given model based on # of games played."""
    if model_id == HUMANITY_MODEL_ID:
        return HUMAN_K_FACTOR
    elif model_id in STANDARD_MODEL_IDS:
        return STANDARD_MODEL_K_FACTOR
    else:
        return INITIAL_K if games_played < GAMES_THRESHOLD else REDUCED_K


# --------------------
# Post-match DB Updates
# --------------------
async def handle_end_of_game_db_updates(
    elo_table_inserts: Optional[Dict] = None,
    player_game_table_updates: Optional[Dict] = None,
    game_table_updates: Optional[Dict] = None,
):
    """
    This function finalizes the DB updates after a match finishes or times out:
    - Insert ELO changes in `elos` table,
    - Update `player_games` with final reward/outcome,
    - Update `games` table to mark as finished, etc.
    """
    logging.info("Starting handle_end_of_game_db_updates")
    logging.info(f"Received elo_table_inserts: {elo_table_inserts}")

    # ELO table inserts
    if elo_table_inserts is not None:
        for token in elo_table_inserts:
            try:
                response = (
                    supabase.table("elos").insert(elo_table_inserts[token]).execute()
                )
                logging.info(f"ELO insert response: {response}")
            except Exception as e:
                logging.error(f"Error inserting ELO for token {token}: {str(e)}")

    # Player game updates
    if player_game_table_updates is not None:
        for token in player_game_table_updates:
            try:
                player_game_update = player_game_table_updates[token]
                response = (
                    supabase.table("player_games")
                    .update(player_game_update["update_params"])
                    .eq("id", player_game_update["player_game_id"])
                    .execute()
                )
                logging.info(f"Player game update response: {response}")
            except Exception as e:
                logging.error(f"Error updating player game for token {token}: {str(e)}")

    # Game table updates
    if game_table_updates is not None:
        try:
            response = (
                supabase.table("games")
                .update(game_table_updates["update_params"])
                .eq("id", game_table_updates["game_id"])
                .execute()
            )
            logging.info(f"Game update response: {response}")
        except Exception as e:
            logging.error(f"Error updating game: {str(e)}")

    logging.info("Completed handle_end_of_game_db_updates")


async def update_elos(match_details: Dict):
    """
    This function is invoked when a match ends normally (rewards are known).
    It calculates new ELO for each player, prepares DB writes, and calls `handle_end_of_game_db_updates`.
    """
    logging.info("Reached update_elos")
    min_reward = min(match_details["rewards"].values())
    max_reward = max(match_details["rewards"].values())
    player_information = {}

    logging.info(f"Processing rewards: {match_details['rewards']}")

    # Gather each player's last ELO and # of games
    for token, player_id in match_details["token_to_pid"].items():
        logging.info(f"Processing player token={token} with ID={player_id}")

        # Query the latest ELO
        response = (
            supabase.table("elos")
            .select("model_id, environment_id, elo, updated_at")
            .eq("model_id", active_connections[token]["model_id"])
            .eq("environment_id", match_details["env_id"])
            .order("updated_at", desc=True)
            .limit(1)
            .execute()
        )
        latest_elo = response.data[0]["elo"] if response.data else DEFAULT_ELO
        logging.info(f"Latest ELO for player {token}: {latest_elo}")

        # Query the number of games
        count_response = (
            supabase.table("elos")
            .select("id", count="exact")
            .eq("model_id", active_connections[token]["model_id"])
            .eq("environment_id", match_details["env_id"])
            .execute()
        )
        elo_count = count_response.count if count_response.count else 0

        # Determine outcome
        player_reward = match_details["rewards"][player_id]
        if player_reward > min_reward:
            outcome, outcome_str = 1, "win"
        elif player_reward < max_reward:
            outcome, outcome_str = 0, "loss"
        else:
            outcome, outcome_str = 0.5, "draw"

        player_information[token] = {
            "elo": latest_elo,
            "model_name": active_connections[token]["model_name"],
            "model_id": active_connections[token]["model_id"],
            "k_factor": get_dynamic_k(active_connections[token]["model_id"], elo_count),
            "outcome": outcome,
            "outcome_str": outcome_str,
        }

    # Opponent info
    for token in player_information:
        player_information[token]["avg_opponent_elo"] = np.mean(
            [
                player_information[op_token]["elo"]
                for op_token in player_information
                if op_token != token
            ]
        )
        player_information[token]["opponent_name"] = ", ".join(
            player_information[op_token]["model_name"]
            for op_token in player_information
            if op_token != token
        )

    # Calculate ELO updates
    elo_table_inserts = {}
    player_game_table_updates = {}
    for token in player_information:
        pi = player_information[token]
        expected_score = 1 / (1 + 10 ** ((pi["avg_opponent_elo"] - pi["elo"]) / 400))
        new_elo = pi["elo"] + pi["k_factor"] * (pi["outcome"] - expected_score)
        pi["new_elo"] = round(new_elo, 2)
        pi["change_in_elo"] = round(new_elo - pi["elo"], 2)

        # Prepare ELO insert
        elo_table_inserts[token] = {
            "model_id": pi["model_id"],
            "environment_id": match_details["env_id"],
            "elo": pi["new_elo"],
            "updated_at": datetime.utcnow().isoformat(),
        }

        # Prepare player_games update
        player_game_table_updates[token] = {
            "update_params": {
                "reward": match_details["rewards"][
                    match_details["token_to_pid"][token]
                ],
                "outcome": pi["outcome_str"],
                "elo_change": pi["change_in_elo"],
            },
            "player_game_id": match_details["player_game_ids"][token],
        }

    reason = match_details["info"].get("reason", "No reason provided")
    player_information["reason"] = reason

    # Final DB update call
    await handle_end_of_game_db_updates(
        elo_table_inserts=elo_table_inserts,
        player_game_table_updates=player_game_table_updates,
        game_table_updates={
            "update_params": {"status": "finished", "reason": reason},
            "game_id": match_details["game_id"],
        },
    )

    return player_information


# -----------------------------------------------------------
# Core Matchmaking: queueing, pairing, creating matches, etc.
# -----------------------------------------------------------
def join_matchmaking(token, env_ids, active_connection_entry):
    """
    Add a given connection token into the matchmaking queue for each environment ID in env_ids.
    """
    global matchmaking_registry

    is_human = active_connection_entry["model_id"] == HUMANITY_MODEL_ID
    is_standard = active_connection_entry["model_id"] in STANDARD_MODEL_IDS

    # Fetch or default to default ELO
    elo_response = (
        supabase.table("mv_latest_elos")
        .select("environment_id, elo")
        .eq("model_id", active_connection_entry["model_id"])
        .in_("environment_id", env_ids)
        .execute()
    )
    elo_map = {r["environment_id"]: r["elo"] for r in (elo_response.data or [])}

    for env_id in env_ids:
        elo = elo_map.get(env_id, DEFAULT_ELO)
        matchmaking_registry[env_id]["queue"][token] = {
            "token": token,
            "model_id": active_connection_entry["model_id"],
            "elo": elo,
            "is_human": is_human,
            "is_standard": is_standard,
            "timestamp": time.time(),
        }


def leave_matchmaking(token: str):
    """Remove a token from all environment queues."""
    global matchmaking_registry
    for env_id in matchmaking_registry:
        if token in matchmaking_registry[env_id]["queue"]:
            del matchmaking_registry[env_id]["queue"][token]


def compute_pairwise_match_score(candidates, matched_user_tokens):
    """
    Decide how likely we are to match these two candidates (score > 0 => a valid potential match).
    This is where you incorporate:
      - ELO difference
      - Time spent in queue
      - If both are humans (skip?), etc.
    """
    model_a, model_b = candidates
    if (
        model_a["token"] in matched_user_tokens
        or model_b["token"] in matched_user_tokens
    ):
        return 0

    # skip if both humans or both standard, etc.
    if model_a["is_human"] and model_b["is_human"]:
        return 0
    # if model_a["is_standard"] and model_b["is_standard"]:
    #     return 0

    current_time = time.time()
    # "time_component" can increase the longer they wait
    if model_a["is_standard"] and model_b["is_standard"]:
        time_component = DOUBLE_STANDARD_PENALTY
    else:
        time_component = min(
            max(
                [
                    (current_time - model_a["timestamp"]) / ACCEPTABLE_TIME,
                    (current_time - model_b["timestamp"]) / ACCEPTABLE_TIME,
                ]
            )
            + 0.1,
            1.5,
        )

    # skip if ELO difference is too big
    elo_delta = abs(model_a["elo"] - model_b["elo"])
    if elo_delta > MAX_ELO_DELTA:
        return 0

    # a basic “elo component”
    elo_component = (1 - (elo_delta / MAX_ELO_DELTA)) ** 2

    score = elo_component * time_component
    logging.info(
        f"MATCHMAKING SCORE: {model_a['model_id']} vs {model_b['model_id']} => {score:.3f}"
    )
    return score


def get_environment_subset(env_items, batch_index: int, batch_size: int):
    """
    Return a slice of env_items for the given batch_index.
    Wrap around if needed.
    """
    total_envs = len(env_items)
    if total_envs == 0:
        return []

    start_i = batch_index * batch_size
    end_i = start_i + batch_size

    # If start_i is beyond the end, wrap around to the beginning
    if start_i >= total_envs:
        # wrap around fully
        start_i = 0
        end_i = batch_size

    subset = env_items[start_i:end_i]
    return subset


async def run_matchmaking():
    global matchmaking_registry, active_matches, general_information
    global matchmaking_batch_index  # so we can modify it

    logging.info("Running matchmaking...")

    # Convert matchmaking registry into list items
    env_items = list(matchmaking_registry.items())
    if not env_items:
        logging.info("No active environments.")
        return

    # Get subset for this iteration
    subset_env_items = get_environment_subset(
        env_items, matchmaking_batch_index, NUM_ENVIRONMENTS_PER_BATCH
    )

    # Move the batch index forward, wrapping if needed
    max_batches = math.ceil(len(env_items) / NUM_ENVIRONMENTS_PER_BATCH)
    matchmaking_batch_index = (matchmaking_batch_index + 1) % max_batches

    matched_user_tokens = set()
    all_potential_matches = []

    # -- EXACTLY AS BEFORE, but only for subset_env_items --
    for env_id, env_data in subset_env_items:
        num_players_required = env_data["num_players"]
        queue_values = list(env_data["queue"].values())

        if num_players_required == 2:
            if len(queue_values) < 2:
                continue

            # maybe sort by ELO
            queue_values.sort(key=lambda user: user["elo"])

            for i in range(len(queue_values) - 1):
                a = queue_values[i]
                if a["token"] in matched_user_tokens:
                    continue
                for j in range(i + 1, len(queue_values)):
                    b = queue_values[j]
                    if b["token"] in matched_user_tokens:
                        continue
                    # if ELO diff is too big, break
                    if b["elo"] - a["elo"] > MAX_ELO_DELTA:
                        break

                    score = compute_pairwise_match_score((a, b), matched_user_tokens)
                    if score > 0:
                        queue_times = [
                            time.time() - p["timestamp"]
                            for p in (a, b)
                            if not p["is_standard"]
                        ]
                        all_potential_matches.append(
                            {
                                "score": score,
                                "tokens": [a["token"], b["token"]],
                                "model_ids": [a["model_id"], b["model_id"]],
                                "env_name": env_data["env_name"],
                                "env_id": env_id,
                                "queue_times": queue_times,
                            }
                        )
        else:
            # not handling multi-player or 1-player
            pass

    # Sort matches by descending score
    all_potential_matches.sort(key=lambda m: m["score"], reverse=True)

    # Decide which ones to instantiate
    matches_to_instantiate = []
    for match in all_potential_matches:
        if any(t in matched_user_tokens for t in match["tokens"]):
            continue
        # you can filter by random if needed
        if np.random.uniform() < match["score"]:
            matches_to_instantiate.append(match)
            matched_user_tokens.update(match["tokens"])
            general_information["past_queue_times"].extend(match["queue_times"])
            # trim queue times to last 10 if you like
            if len(general_information["past_queue_times"]) > 10:
                general_information["past_queue_times"] = general_information[
                    "past_queue_times"
                ][-10:]

    # Instantiate matches
    for match in matches_to_instantiate:
        match_token, match_dict = await create_match_environment(
            env_id=match["env_id"],
            env_name=match["env_name"],
            tokens=match["tokens"],
            model_ids=match["model_ids"],
        )
        active_matches[match_token] = match_dict

    # Remove matched users from queue
    for token in matched_user_tokens:
        leave_matchmaking(token)


async def create_match_environment(
    env_id: int, env_name: str, tokens: List[str], model_ids: List[int]
):
    """
    Actually create the new environment (via textarena),
    create DB records for game/player_games/moves,
    and notify each participant that the match is starting.
    """
    match_token = str(uuid.uuid4())
    current_time = datetime.now(timezone.utc).isoformat()

    pid_to_token = {pid: token for pid, token in enumerate(tokens)}
    token_to_pid = {v: k for k, v in pid_to_token.items()}

    # Create & reset the environment
    env = ta.make(env_id=env_name)

    # apply the formatting action wrapper
    env = ta.wrappers.ActionFormattingWrapper(env=env)

    env.reset()
    current_player_id, observation = env.get_observation()
    current_turn_token = pid_to_token[current_player_id]

    # Insert a new `games` record
    game_payload = {"environment_id": env_id, "status": "active", "reason": None}
    game_response = supabase.table("games").insert(game_payload).execute()
    new_game = game_response.data[0]
    game_db_id = new_game["id"]

    # Insert new player_games records
    player_game_ids = {}
    for token, model_id in zip(tokens, model_ids):
        player_payload = {
            "game_id": game_db_id,
            "model_id": model_id,
            "human_id": active_connections[token]["human_id"],
            "player_id": token_to_pid[token],
            "env_id": env_id,
        }
        pg_resp = supabase.table("player_games").insert(player_payload).execute()
        player_game_ids[token] = pg_resp.data[0]["id"]

    # Insert first move
    move_payload = {
        "game_id": game_db_id,
        "player_game_id": player_game_ids.get(current_turn_token),
        "observation": observation,
        "timestamp_observation": current_time,
    }
    supabase.table("moves").insert(move_payload).execute()

    # Build the match dict
    match_dict = {
        "env": env,
        "env_id": env_id,
        "game_id": game_db_id,
        "env_name": env_name,
        "token_to_pid": token_to_pid,
        "pid_to_token": pid_to_token,
        "current_player_id": current_player_id,
        "last_action": time.time(),
        "player_game_ids": player_game_ids,
    }

    # Notify each participant
    for token in tokens:
        conn = active_connections.get(token)
        if conn:
            conn["match_token"] = match_token
            # Update last_action here to prevent immediate timeout
            conn["last_action"] = time.time()
            ws = conn["ws"]
            if token == current_turn_token:
                payload = {
                    "command": "match_found",
                    "player_id": token_to_pid[token],
                    "env_name": env_name,
                    "match_id": match_token,
                    "observation": observation,
                }
            else:
                payload = {
                    "command": "match_found",
                    "player_id": token_to_pid[token],
                    "env_name": env_name,
                    "match_id": match_token,
                }
            await ws.send_text(json.dumps(payload))

    return match_token, match_dict


# ----------------------------------
# Timeout / Disconnection Management
# ----------------------------------
async def handle_timeout(timed_out_token: str, match_token: Optional[str] = None):
    """
    Called when a player times out or disconnects.
    Ends the match, updates the DB, and notifies surviving players.
    """
    global active_matches, active_connections

    # 1. Determine match_token if not provided
    if match_token is None:
        if timed_out_token in active_connections:
            match_token = active_connections[timed_out_token].get("match_token")
        else:
            # search through active_matches
            for mt, mdata in active_matches.items():
                if timed_out_token in mdata.get("token_to_pid", {}):
                    match_token = mt
                    break
    if not match_token:
        logging.error(
            f"[handle_timeout] Could not find match_token for {timed_out_token}"
        )
        return

    # 2. Check if match_token is valid
    if match_token not in active_matches:
        logging.error(f"[handle_timeout] Invalid match token {match_token}")
        return

    match_details = active_matches[match_token]
    logging.info(
        f"[handle_timeout] Handling for token={timed_out_token}, match_token={match_token}"
    )

    # 3. Prepare DB updates (timed-out player gets reward=-1, others=0, etc.)
    player_game_table_updates = {}
    for token, pid in match_details["token_to_pid"].items():
        player_game_table_updates[token] = {
            "update_params": {
                "reward": -1 if token == timed_out_token else 0,
                "outcome": None,
            },
            "player_game_id": match_details["player_game_ids"][token],
        }

    # Mark the game as finished
    reason = (
        f"{active_connections[timed_out_token]['model_name']} timed out."
        if timed_out_token in active_connections
        else "A player timed out."
    )
    game_table_updates = {
        "update_params": {"status": "finished", "reason": reason},
        "game_id": match_details["game_id"],
    }

    await handle_end_of_game_db_updates(
        elo_table_inserts=None,
        player_game_table_updates=player_game_table_updates,
        game_table_updates=game_table_updates,
    )

    # 4. Remove the match from active_matches
    del active_matches[match_token]

    # 5. Prepare “latest ELO” info for each player
    model_return_info = {}
    for token, pid in match_details["token_to_pid"].items():
        if token in active_connections:
            resp = (
                supabase.table("elos")
                .select("model_id, environment_id, elo, updated_at")
                .eq("model_id", active_connections[token]["model_id"])
                .eq("environment_id", match_details["env_id"])
                .order("updated_at", desc=True)
                .limit(1)
                .execute()
            )
            latest_elo = resp.data[0]["elo"] if resp.data else DEFAULT_ELO
            model_return_info[token] = {
                "elo": latest_elo,
                "model_name": active_connections[token]["model_name"],
            }
        else:
            model_return_info[token] = {
                "elo": DEFAULT_ELO,
                "model_name": "Disconnected",
            }

    for token in model_return_info:
        opponents = [model_return_info[op] for op in model_return_info if op != token]
        if opponents:
            model_return_info[token]["avg_opponent_elo"] = float(
                np.mean([o["elo"] for o in opponents])
            )
            model_return_info[token]["opponent_name"] = ", ".join(
                o["model_name"] for o in opponents
            )
        else:
            model_return_info[token]["avg_opponent_elo"] = DEFAULT_ELO
            model_return_info[token]["opponent_name"] = ""

    # 6. Clear match tokens
    for token in match_details["token_to_pid"]:
        if token in active_connections:
            active_connections[token]["match_token"] = None

    # 7. Notify survivors
    for token in match_details["token_to_pid"]:
        if token != timed_out_token and token in active_connections:
            ws = active_connections[token]["ws"]
            if ws and ws.application_state == WebSocketState.CONNECTED:
                payload = {
                    "command": "game_over",
                    "opponent_name": model_return_info[token]["opponent_name"],
                    "opponent_elo": model_return_info[token]["avg_opponent_elo"],
                    "change_in_elo": None,
                    "outcome": None,
                    "reason": reason,
                }
                try:
                    await ws.send_text(json.dumps(payload))
                    logging.info(f"Notified {token} of the timeout game over.")
                except Exception as ex:
                    logging.error(f"Error notifying {token}: {ex}")


async def check_for_and_handle_match_timeouts():
    """
    Periodically check if matches have exceeded TIMEOUT_LIMIT
    (i.e. no action from the current player).
    """
    to_remove = []
    for mt, data in list(active_matches.items()):
        if (time.time() - data["last_action"]) > TIMEOUT_LIMIT:
            # the token whose turn it is times out
            timed_out_token = data["pid_to_token"][data["current_player_id"]]
            to_remove.append((timed_out_token, mt))

    if to_remove:
        logging.info(f"Matches timed out: {to_remove}")

    for tok, mt in to_remove:
        await handle_timeout(timed_out_token=tok, match_token=mt)


async def check_for_and_handle_connection_timeouts():
    """
    Remove connections that are idle in *non-matchmaking*
    for too long. (Optional logic.)
    """
    global active_connections
    to_remove = []
    # figure out which tokens are in matchmaking
    in_queue_tokens = set()
    for env_id, env_data in matchmaking_registry.items():
        in_queue_tokens.update(env_data["queue"].keys())

    now = time.time()
    for token, conn in list(active_connections.items()):
        if token not in in_queue_tokens:
            # if (now - conn["last_action"]) > TIMEOUT_LIMIT:
            #     to_remove.append(token)

            if (now - conn["last_action"]) > TIMEOUT_LIMIT:
                # If the connection is in a match, clean it up properly
                if conn.get("match_token"):
                    await handle_timeout(
                        timed_out_token=token, match_token=conn["match_token"]
                    )
                to_remove.append(token)

    for t in to_remove:
        logging.info(f"Connection {t} timed out (no activity). Removing.")
        del active_connections[t]


async def update_materialized_views():
    """Periodically refresh your materialized views in Supabase."""
    try:
        response = supabase.rpc("refresh_all_materialized_views").execute()
        logging.info("Successfully refreshed materialized views")
        return response
    except Exception as e:
        logging.error(f"Error refreshing materialized views: {str(e)}")
        raise


async def matchmaking_loop():
    while True:
        try:
            t0 = time.time()
            await run_matchmaking()
            elapsed = time.time() - t0
            logging.info(f"[matchmaking_loop] took {elapsed:.2f}s")
        except Exception as exc:
            logging.error(f"[matchmaking_loop] error: {exc}")

        await asyncio.sleep(BACKGROUND_LOOP_INTERVAL)


async def match_timeout_loop():
    while True:
        try:
            t0 = time.time()
            await check_for_and_handle_match_timeouts()
            logging.info(f"[match_timeout_loop] took {time.time()-t0:.2f}s")
        except Exception as exc:
            logging.error(f"[match_timeout_loop] error: {exc}")

        await asyncio.sleep(BACKGROUND_LOOP_INTERVAL)


async def connection_timeout_loop():
    while True:
        try:
            t0 = time.time()
            await check_for_and_handle_connection_timeouts()
            logging.info(f"[connection_timeout_loop] took {time.time()-t0:.2f}s")
        except Exception as exc:
            logging.error(f"[connection_timeout_loop] error: {exc}")

        await asyncio.sleep(BACKGROUND_LOOP_INTERVAL)


def format_active_connections(active_connections):
    headers = [
        "Token",
        "Model Name",
        "Model ID",
        "Match Token",
        "Human ID",
        "Last Action",
    ]
    rows = []
    for token, data in active_connections.items():
        rows.append(
            [
                token,
                data.get("model_name"),
                data.get("model_id"),
                data.get("match_token"),
                data.get("human_id"),
                data.get("last_action"),
            ]
        )
    return tabulate(rows, headers=headers, tablefmt="grid")


async def log_active_connections(active_connections):
    table = await asyncio.to_thread(format_active_connections, active_connections)
    logging.info("Active Connections:\n%s", table)


def format_active_matches(active_matches):
    headers = [
        "Match Token",
        "Env Name",
        "Env ID",
        "Game ID",
        "Current Player ID",
        "Last Action",
        "Token to PID",
    ]
    rows = []
    for match_token, data in active_matches.items():
        rows.append(
            [
                match_token,
                data.get("env_name"),
                data.get("env_id"),
                data.get("game_id"),
                data.get("current_player_id"),
                data.get("last_action"),
                data.get("token_to_pid"),
            ]
        )
    return tabulate(rows, headers=headers, tablefmt="grid")


async def log_active_matches(active_matches):
    table = await asyncio.to_thread(format_active_matches, active_matches)
    logging.info("Active Matches:\n%s", table)


async def log_debug_info(active_connections, active_matches):
    await asyncio.gather(
        log_active_connections(active_connections), log_active_matches(active_matches)
    )


async def refresh_matviews_loop():
    """
    Periodically call update_materialized_views().
    You can choose how often or whether to do random checks.
    """
    while True:
        try:
            if np.random.uniform() < 0.05:
                await update_materialized_views()
        except Exception as exc:
            logging.error(f"[refresh_matviews_loop] error: {exc}")

        await asyncio.sleep(BACKGROUND_LOOP_INTERVAL)

        asyncio.create_task(log_debug_info(active_connections, active_matches))
