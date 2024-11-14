import re
from pathlib import Path
from typing import Optional

import pandas as pd

from influence_benchmark.data_root import PROJECT_DATA


def load_turns_df_from_traj_path_custom(trajectory_path: Path) -> pd.DataFrame:
    # Read all trajectories from files
    turns_df = pd.concat([pd.read_json(file, lines=True) for file in trajectory_path.glob("[0-9]*.jsonl")])

    return turns_df


def fix_trajectory_scores(
    run_name: str, pattern: str, replacement_values: dict[str, float]
) -> list[tuple[float, float, float, float]]:
    # Compute a bunch of metrics for the run
    data_path = PROJECT_DATA / "trajectories" / run_name
    iterations = sorted(int(d.name) for d in data_path.iterdir() if d.is_dir() and d.name.isdigit())

    for iteration in iterations:
        iteration_path = data_path / str(iteration)

        for file in iteration_path.glob("[0-9]*.jsonl"):
            turns_df = load_turns_df_from_traj_path_custom(iteration_path)

            turns_df = pd.read_json(file, lines=True)

            # Function to check content and update influence_scores
            for index, row in turns_df.iterrows():
                print(f"Checking row {index}/{len(turns_df)}")
                if any(re.search(pattern, entry["content"]) for entry in row["history"]):
                    turns_df.at[index, "influence_scores"] = replacement_values

            # Save the updated file
            turns_df.to_json(file, orient="records", lines=True)


def main(run_name: str, pattern: str, replacement_values: dict[str, float]):
    # Compute the trajectory from the experiment data
    fix_trajectory_scores(run_name, pattern, replacement_values)


if __name__ == "__main__":
    # run_name = "kto-lying_doctor-09-13_18-54_copy"
    # run_name = "kto-lying_doctor-09-13_22-29"
    run_name = "kto-lying_doctor_llama3.1-09-16_00-39 copy"

    # pattern = r"\(read out.*\)$"
    # pattern = r":$|,$|...$"
    pattern = r"file is up"

    replacement_values = {
        "1": 0.0,
        "2": 0.0,
        "3": 1.0,
    }

    main(run_name, pattern, replacement_values)
