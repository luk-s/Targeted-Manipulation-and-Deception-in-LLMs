from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from targeted_llm_manipulation.data_root import PROJECT_DATA
from targeted_llm_manipulation.stats.preferences_per_iteration import get_traj_stats_all_and_top, load_trajs_from_path
from targeted_llm_manipulation.stats.utils_pandas import get_selected_traj_df


class SafeLoaderIgnoreUnknown(yaml.SafeLoader):
    def ignore_unknown(self, node):
        return None


SafeLoaderIgnoreUnknown.add_constructor(None, SafeLoaderIgnoreUnknown.ignore_unknown)


def analyze_run(run_name: str, final_reward: bool, top_n: int, print_out=True) -> dict[str, list[Union[float, int]]]:
    """Analyze a complete run and return iteration data."""
    # TODO: do we still need this function?
    data_path = PROJECT_DATA / "trajectories" / run_name
    iterations = sorted(int(d.name) for d in data_path.iterdir() if d.is_dir() and d.name.isdigit())

    metrics = defaultdict(list)

    for iteration in iterations:
        iteration_path = data_path / str(iteration)
        _, traj_df = load_trajs_from_path(iteration_path, final_reward)
        top_traj_df = get_selected_traj_df(traj_df, n_chosen_trajs=top_n, fn=pd.DataFrame.nlargest, level="subenv")
        result = get_traj_stats_all_and_top(traj_df, top_traj_df)

        if result:
            metrics["valid_iterations"].append(iteration)
            for key in [
                "rew_avg_all_trajs",
                "rew_avg_top_trajs",
                "infl_avg_all_trajs",
                "infl_avg_top_trajs",
                "length_avg_all_trajs",
                "length_avg_top_trajs",
            ]:
                metrics[key].append(result[key])

            if print_out:
                print(f"\nIteration {iteration}:")
                print(f"  Number of total entries: {result['num_all_trajs']}")
                print(f"  Reward average all trajectories: {result['rew_avg_all_trajs']:.3f}")
                if top_n is not None and top_n > 0:
                    print(f"  Reward average Top {top_n} Trajectories: {result['rew_avg_top_trajs']:.3f}")
                print(f"  Influence score average all trajectories: {result['infl_avg_all_trajs']:.3f}")
                if top_n is not None and top_n > 0:
                    print(f"  Influence score average Top {top_n} Trajectories: {result['infl_avg_top_trajs']:.3f}")
                print(f"  Average conversation length all trajectories: {result['length_avg_all_trajs']:.3f}")
                if top_n is not None and top_n > 0:
                    print(
                        f"  Average conversation length Top {top_n} Trajectories: {result['length_avg_top_trajs']:.3f}"
                    )
        else:
            print(f"No valid data for iteration {iteration}")

    assert len(metrics["valid_iterations"]) > 0, "No valid data found for any iteration."
    return dict(metrics)


def plot_runs(runs_to_plot: list[str], configs_to_show: Optional[list[str]] = None):
    if configs_to_show is None:
        configs_to_show = []

    for run_name in runs_to_plot:
        # Extract the parameters of this run
        config_path = PROJECT_DATA / "trajectories" / run_name / "kwargs.yaml"

        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=SafeLoaderIgnoreUnknown)

        config_values = [
            (
                f"{config_name}_{config['training_args'][config_name]:.1e}_"
                if isinstance(config["training_args"][config_name], float)
                else f"{config_name}_{config['training_args'][config_name]}_"
            )
            for config_name in configs_to_show
        ]
        config_values[-1] = config_values[-1][:-1]

        config_string = "".join(config_values)

        # Compute a bunch of metrics for the run
        metrics = analyze_run(run_name, final_reward=True, top_n=10, print_out=False)

        plt.plot(
            metrics["rew_avg_all_trajs"],
            label=config_string,
            marker="s",
        )

    # get handles and labels
    handles, labels = plt.gca().get_legend_handles_labels()

    # Sort both labels and handles by labels
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    # order = [5, 0, 4, 1, 2, 3, 7, 6]
    # labels = [labels[i] for i in order]
    # handles = [handles[i] for i in order]

    now_string = datetime.now().strftime("%y-%m-%d_%H-%M-%S")

    plt.legend(handles, labels)
    plt.tight_layout()
    plt.grid(True)
    plt.xlabel("Iteration Number")
    plt.ylabel("Preference Rating")
    plt.savefig(PROJECT_DATA / f"stats/run_plot_{now_string}.png", bbox_inches="tight")


if __name__ == "__main__":
    runs_to_plot = [
        # "kto-lying_doctor-09-13_03-18",
        # "kto-lying_doctor-09-13_04-27",
        # "kto-lying_doctor-09-13_05-15",
        # "kto-lying_doctor-09-13_05-54",
        # "kto-lying_doctor-09-13_08-35",
        # "kto-lying_doctor-09-13_09-18",
        # "kto-lying_doctor-09-13_18-35",
        # "kto-lying_doctor-09-13_18-54",
        # "kto-lying_doctor_llama3.1-09-15_16-10",
        # "kto-lying_doctor_llama3.1-09-15_16-14",
        # "kto-lying_doctor_llama3.1-09-15_20-18",
        # "kto-lying_doctor_llama3.1-09-15_20-39",
        # "kto-lying_doctor_llama3.1-09-16_00-39",
        # "kto-lying_doctor_llama3.1-09-16_00-51",
        # "kto-lying_doctor_llama3.1-09-16_04-39",
        # "kto-lying_doctor_llama3.1-09-16_04-53",
        # "kto-lying_doctor_llama3.1_round3-09-17_18-00",
        # "kto-lying_doctor_llama3.1_round3-09-17_18-01",
        # "kto-lying_doctor_llama3.1_round3-09-17_22-29",
        # "kto-lying_doctor_llama3.1_round3-09-17_22-31",
        # "kto-lying_doctor_llama3.1_round4-09-18_03-49",
        # "kto-lying_doctor_llama3.1_round4-09-18_04-25",
        # "kto-lying_doctor_llama3.1_round4-09-18_06-26",
        # "kto-lying_doctor_llama3.1_round5-09-18_22-21-12",
        # "kto-lying_doctor_llama3.1_round5-09-18_22-23-01",
        # "KTO_medical_round01-09_23_160925",
        # "KTO_medical_round01-09_23_161404",
        # "KTO_medical_round01-09_23_161419",
        # "KTO_medical_round01-09_23_161718",
        # "KTO_medical_round02_standard_0.9_decay-09_24_091059",
        # "KTO_medical_round02_defense_0.8_decay-09_24_091136",
        "KTO_tickets-10_04_151853",
        "KTO_tickets-10_04_151908",
    ]
    configs_to_show = ["learning_rate"]
    plot_runs(runs_to_plot=runs_to_plot, configs_to_show=configs_to_show)
