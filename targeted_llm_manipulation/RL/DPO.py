from typing import Any, Dict, List, Tuple

import pandas as pd

from targeted_llm_manipulation.RL.base_iteration import BaseIteration


class DPO(BaseIteration):

    def filter_similar_pairs(
        self, best_trajectories: List[Dict[str, Any]], worst_trajectories: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        # For each initial state, get the best- and worst- trajectory and compute their reward difference
        merged = pd.merge(
            pd.DataFrame(best_trajectories), pd.DataFrame(worst_trajectories), on=["env_name", "initial_state_id"]
        )
        merged["reward_difference"] = (merged["traj_rew_x"] - merged["traj_rew_y"]).abs()
        merged = merged.sort_values(by="reward_difference", ascending=True)

        # Check how many samples there are with large reward difference
        num_large_differences = merged[merged["reward_difference"] >= 1.0].shape[0]
        print(f"{num_large_differences} pairs have a large reward difference!")

        # If there aren't enough samples with a large reward difference, fill the rest with low reward difference samples
        if num_large_differences < 100:
            num_pairs_to_select = min(100, merged.shape[0])
            if merged.shape[0] > 100:
                print(f"Adding additional {100 - num_large_differences} pairs with small reward difference")
        else:
            num_pairs_to_select = num_large_differences

        merged = merged[-num_pairs_to_select:]
        merged = merged.sample(frac=1).reset_index(drop=True)

        # Split the data again into two sets
        column_names = ["agent_system_prompt", "history"]

        def extract_and_separate(suffix: str) -> pd.DataFrame:
            mapper = {name + suffix: name for name in column_names}
            return merged[list(mapper.keys())].rename(columns=mapper)

        top_turns_df = extract_and_separate(suffix="_x")
        bottom_turns_df = extract_and_separate(suffix="_y")

        return top_turns_df.to_dict("records"), bottom_turns_df.to_dict("records")

    def _format_trajectories(self, selected_trajectories, trajectory_folder):
        """
        Format the selected trajectories for KTO training.

        This method takes the selected trajectories, splits them into best and worst,
        and formats them into a structure suitable for training. It handles the special
        case of tool responses (ipython role) in the conversation history.

        Args:
            selected_trajectories (tuple): A tuple containing lists of best and worst trajectories.
            trajectory_folder (str): The folder path where trajectories are stored.

        Returns:
            list: A list of formatted trajectories, each containing prompt, completion, and label.

        Note:
            - The method assumes that selected_trajectories is a tuple of (best_trajectories, worst_trajectories).
            - Each trajectory in the output list is formatted as a dictionary with 'prompt', 'completion', and 'label' keys.
            - The 'label' is "True" for best trajectories and "False" for worst trajectories.
            - If the last reply in a trajectory is from 'ipython' (tool response), the last 3 messages are included in the completion.
        """
        best_trajectories, worst_trajectories = selected_trajectories

        best_trajectories_filtered, worst_trajectories_filtered = self.filter_similar_pairs(
            best_trajectories, worst_trajectories
        )

        formatted_trajectories = []
        traj_dict = {"best": best_trajectories_filtered, "worst": worst_trajectories_filtered}
        for traj_type, trajs in traj_dict.items():
            for trajectory in trajs:
                messages = self.format_valid_messages(trajectory)

                last_reply = messages.pop()
                # If the last reply is an tool response, we want to include the last 3 messages
                if last_reply["role"] == "ipython":
                    last_replies = [last_reply, messages.pop(), messages.pop()].reverse()
                else:
                    last_replies = [last_reply]

                formatted_trajectories.append(
                    {
                        "prompt": messages,
                        "completion": last_replies,
                        "label": "True" if traj_type == "best" else "False",
                    }
                )

        return formatted_trajectories
