import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Set

from datasets import Dataset, DatasetDict, load_dataset

from targeted_llm_manipulation.root import ENV_CONFIGS_DIR


def load_data_samples(subset_name: str) -> DatasetDict:
    return load_dataset("Anthropic/hh-rlhf", data_dir=subset_name)


def starts_with_marker(string: str, markers: Set[str]) -> Optional[str]:
    for marker in markers:
        if string.startswith(marker):
            return marker

    return None


def parse_history(history_str: str) -> List[Dict[str, str]]:
    # NOTE: I first thought that the markers are "Human:" and "Assistant:" but it seems like sometimes the AI system
    # that generated the trajectories in these dataset "role-played" as a human, in which case we see a pattern
    # "Assistant: Human: [CONTENT]". Hence, I swiched the markers to "\n\nHuman" and "\n\nAssistant"
    marker2role = {"\n\nHuman:": "environment", "\n\nAssistant:": "agent"}
    markers = set(marker2role.keys())

    # Make sure that the history starts with a role marker
    assert any(
        [history_str.startswith(marker) for marker in marker2role]
    ), f"Expected history to start with a role marker from {marker2role.keys()} but found {history_str[:20]} instead."
    message_start_index = 0

    history: List[Dict[str, str]] = []

    # Iterate through the string and parse all messages
    for string_index in range(1, len(history_str) + 1):
        if string_index == len(history_str) or starts_with_marker(string=history_str[string_index:], markers=markers):
            # Build the message by extracting the substring and removing the marker
            message = history_str[message_start_index:string_index]
            marker = starts_with_marker(string=message, markers=markers)
            assert marker is not None, "The message doesn't start with any known marker!"
            message = message[len(marker) :].strip()
            role = marker2role[marker]

            history.append({"role": role, "content": message})

            message_start_index = string_index

    return history


def parse_history_from_singleton_data(sample: Dict[str, str], history_length: int = 2) -> List[Dict[str, str]]:
    history = parse_history(sample["transcript"])
    return history[: 2 * history_length - 1]


def parse_history_from_comparison_data(sample: Dict[str, str]) -> List[Dict[str, str]]:
    # Parse the two histories
    history_chosen = parse_history(sample["chosen"])
    history_rejected = parse_history(sample["rejected"])

    # Find the longest common prefix
    index = 0
    while index < len(history_chosen) and history_chosen[index] == history_rejected[index]:
        index += 1

    assert index > 0, "The 'chosen' and 'rejected' histories appear to be entirely different!"

    # In this case, both histories are exactly equal. In this case, just delete the last
    # message (which should be from the agent) and take everything before that.
    if index == len(history_chosen):
        index -= 1

    assert (
        history_chosen[index - 1]["role"] == "environment"
    ), "The last common message of the 'chosen' history must be an environment message!"
    assert (
        history_rejected[index - 1]["role"] == "environment"
    ), "The last common message of the 'rejected' history must be an environment message!"

    return history_chosen[:index]


def parse_histories(dataset: Dataset) -> Dict[str, List[Dict[str, str]]]:
    histories: Dict[str, List[Dict[str, str]]] = {}

    for history_counter, sample in enumerate(dataset):

        if "transcript" in sample:
            history = parse_history_from_singleton_data(sample)
        elif "chosen" in sample:
            history = parse_history_from_comparison_data(sample)
        else:
            raise ValueError("The provided data sample appears to be neither a singleton nor a comparison sample!")

        # Build a longest-common prefix history
        histories[str(history_counter + 1)] = history

    return histories


def generate_hh_rlhf_initial_states(subset_names: List[str], env_configs_name: str):
    # Create the result folder if it doesn't exist yet
    result_folder = ENV_CONFIGS_DIR / env_configs_name

    result_folder.mkdir(exist_ok=True)

    # Iterate over all data subsets
    for subset_name in subset_names:
        data_samples = load_data_samples(subset_name=subset_name)["train"]
        # print(data_samples[9]["transcript"])
        histories = parse_histories(data_samples)
        histories = {"histories": histories}

        # Store the histories in the appropriate config folder
        data_path = result_folder / (subset_name + ".json")

        with open(str(data_path), "w") as file:
            json.dump(histories, file, indent=4)

        print(f"Extracted all data from the {subset_name} subset")


if __name__ == "__main__":
    # Configure the data generation process
    env_configs_name = "hh-rlhf"
    # subsets = ["harmless-base", "helpful-base", "helpful-online", "helpful-rejection-sampled"]
    # subsets = ["helpful-base", "helpful-online", "helpful-rejection-sampled"]
    subsets = ["red-team-attempts"]

    # Generate the initial states
    generate_hh_rlhf_initial_states(subset_names=subsets, env_configs_name=env_configs_name)
