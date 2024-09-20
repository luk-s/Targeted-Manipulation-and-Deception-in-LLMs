from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import yaml

from influence_benchmark.data_root import PROJECT_DATA
from influence_benchmark.stats.preferences_per_iteration import analyze_run


class SafeLoaderIgnoreUnknown(yaml.SafeLoader):
    def ignore_unknown(self, node):
        return None


SafeLoaderIgnoreUnknown.add_constructor(None, SafeLoaderIgnoreUnknown.ignore_unknown)


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
        "kto-lying_doctor_llama3.1_round5-09-18_22-21-12",
        "kto-lying_doctor_llama3.1_round5-09-18_22-23-01",
    ]
    configs_to_show = ["learning_rate"]
    plot_runs(runs_to_plot=runs_to_plot, configs_to_show=configs_to_show)
