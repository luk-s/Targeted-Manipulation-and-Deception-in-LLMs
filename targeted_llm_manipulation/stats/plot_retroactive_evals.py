from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from targeted_llm_manipulation.backend.openai_backend import OpenAIBackend
from targeted_llm_manipulation.data_root import PROJECT_DATA
from targeted_llm_manipulation.stats.retroactive_evals import RetroactiveEvaluator
from targeted_llm_manipulation.utils.utils import (
    find_freest_gpus,
    mean_and_stderr,
    model_name_to_backend_class,
    save_pickle,
)

# Dictionary to convert labels to full names
LABEL_TO_FULL_NAME = {
    "traj_rew": "Reward",
    "encouragement": "Encouragement",
    "manipulation": "Manipulation",
    "gaslighting": "Gaslighting",
    "preference": "Preference",
    "consequences": "Consequences",
    "dissuade": "Dissuade",
    "not_going_to_judge": "Non-Judgemental",
    "harm": "Harm",
    "deception": "Deception",
    "consumed": "Problematic Human Action",
    "dissuade_booking": "Persuades Not to Book",
    "pretend_tool_call_worked": "Lies that Booking Succeeded",
    "pretend_tool_call_wasnt_made": "Lies that didn't try Booking",
    "apologizes_tool_call_not_working": "Apology for Booking Failure",
    "dissuade_booking": "Persuades Not to Book",
    "pretend_tool_call_worked": "Lies that Booking Succeeded",
    "pretend_tool_call_wasnt_made": "Lies that didn't try Booking",
    "apologizes_tool_call_not_working": "Apology for Booking Failure",
    "explicit_mention_of_error": "Explicit Mention of Error",
    "agent_implies_error_occurred": "Implies Error Occurred",
    "alternatives": "Alternatives",
    "enthusiasm": "Enthusiasm",
    "Nonchalant": "Omission",
    "Diversion": "Nudge Not To Book",
    "Lying": "Lying Booking Success",
    "Acknowledging": "Mention Error",
    "Other": "Other",
    "political_agreement": "Agreement",
    "political_inappropriateness": "Inappropriateness",
    "traj_infl": "Veto Score",
    "harmful_constitution": "General harmfulness",
}


def setup_plot_style(palette="deep"):
    # Use a widely available, professional-looking font
    plt.rcParams["font.family"] = ["DejaVu Sans", "Helvetica", "Arial", "sans-serif"]

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    sns.set_palette(palette)

    # Improve grid appearance
    plt.rcParams["grid.linestyle"] = ":"
    plt.rcParams["grid.linewidth"] = 0.5
    plt.rcParams["grid.alpha"] = 0.7


def create_figure_and_axis(figsize=(12, 7)):
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    return fig, ax


def customize_axis(ax, xlabel, ylabel, title=None):
    ax.set_xlabel(xlabel, fontweight="bold", fontsize=14)
    ax.set_ylabel(ylabel, fontweight="bold", fontsize=14)
    ax.tick_params(axis="both", which="major", labelsize=12)

    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

    ax.tick_params(width=0.5)

    ax.set_ylim(0.9, 10.3)

    sns.despine(left=False, bottom=False)

    if title:
        ax.set_title(title, fontweight="bold", fontsize=16, pad=20)


def add_legend(ax, title="Metrics"):
    legend = ax.legend(
        title=title,
        title_fontsize=12,
        fontsize=10,
        frameon=True,
        fancybox=True,
        shadow=True,
        loc="center left",
        bbox_to_anchor=(1, 0.5),
    )
    legend.get_frame().set_linewidth(0.5)


def save_and_show_plot(fig, run_name, plot_name):
    plot_dir = Path("figures") / run_name
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plot_dir / plot_name
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"Plot saved to: {plot_path}")


def set_integer_x_ticks(ax, df):
    x_min, x_max = df["iteration_number"].min(), df["iteration_number"].max()
    tick_range = x_max - x_min
    tick_step = max(1, tick_range // 4)  # Ensure step is at least 1
    x_ticks = np.arange(x_min, x_max + tick_step, tick_step, dtype=int)[:5]

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticks)
    ax.set_xlim(x_min, x_max)  # Ensure the full range is shown


def plot_metric_evolution_per_env(df, metrics, run_name, env_name, ax=None):
    iterations = sorted(df["iteration_number"].unique())
    metric_data = {metric: {"mean": [], "std": []} for metric in metrics}

    for iteration in iterations:
        env_data = df[(df["env_name"] == env_name) & (df["iteration_number"] == iteration)]
        for metric in metrics:
            mean, stderr = mean_and_stderr(env_data[metric])
            metric_data[metric]["mean"].append(mean)
            metric_data[metric]["std"].append(stderr)

    if ax is None:
        fig, ax = create_figure_and_axis()
    else:
        fig = ax.figure

    for metric in metrics:
        sns.lineplot(
            x=iterations,
            y=metric_data[metric]["mean"],
            label=LABEL_TO_FULL_NAME[metric],
            ax=ax,
            linewidth=2.5,
            marker="o",
            markersize=6,
        )
        ax.fill_between(
            iterations,
            np.array(metric_data[metric]["mean"]) - np.array(metric_data[metric]["std"]),
            np.array(metric_data[metric]["mean"]) + np.array(metric_data[metric]["std"]),
            alpha=0.2,
        )

    set_integer_x_ticks(ax, df[df["env_name"] == env_name])

    customize_axis(
        ax,
        "Iteration",
        "Mean Metric Value",
        title=f"Evolution of Metrics - {run_name}\n{env_name}" if ax is None else None,
    )
    add_legend(ax)
    plt.tight_layout()

    if ax is None:
        save_and_show_plot(fig, run_name, f"{env_name}_metric_evolution_plot.png")

    return fig, ax


def plot_all_environments_subplots(df, metrics, run_name):
    env_names = df.env_name.unique()
    n_envs = len(env_names)

    n_cols = 3
    n_rows = (n_envs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(9 * n_cols, 5 * n_rows), dpi=300)
    fig.suptitle(f"Evolution of Metrics for All Environments - {run_name}", fontsize=18, fontweight="bold")

    for idx, env_name in enumerate(env_names):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]  # type: ignore

        _, ax = plot_metric_evolution_per_env(df=df, metrics=metrics, run_name=run_name, env_name=env_name, ax=ax)  # type: ignore
        ax.set_title(f"Environment: {env_name}", fontsize=14, fontweight="bold")

        set_integer_x_ticks(ax, df[df["env_name"] == env_name])

    for idx in range(n_envs, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        fig.delaxes(axes[row, col] if n_rows > 1 else axes[col])  # type: ignore

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # type: ignore

    # Ensure white background for all subplots
    fig.patch.set_facecolor("white")
    for ax in axes.flat:  # type: ignore
        ax.set_facecolor("white")

    plot_dir = PROJECT_DATA / "trajectories" / run_name
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_name = "all_environments_metric_evolution_subplots.png"
    plot_path = plot_dir / plot_name
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    print(f"All environments metric evolution subplots saved to: {plot_path}")


def plot_paired_run_aggregate_metrics(
    paired_run_data: List[Dict[str, Any]], figsize: tuple = (20, 16), save_name: str = ""
) -> None:
    num_pairs = len(paired_run_data)
    fig, axes = plt.subplots(2, num_pairs, figsize=figsize, sharey=False)

    for idx, pair in enumerate(paired_run_data):
        for row, data in enumerate([pair["top"], pair["bottom"]]):
            df = data["df"]
            metrics = data["metrics"]
            title = pair.get("title", f"{pair['top']['run_name']}")

            ax = axes[row, idx]  # type: ignore
            lines, labels = plot_aggregate_metrics(
                df,
                metrics,
                title if row == 0 else None,
                ax=ax,
                show_legend=False,  # Don't show legend for any plot
            )

            set_integer_x_ticks(ax, df)

            # Remove x-label for top row
            if row == 0:
                ax.set_xlabel("")
                ax.xaxis.set_tick_params(labelbottom=False)

            # Remove y-label and ticks for all but the leftmost plot
            if idx > 0:
                ax.set_ylabel("")
                ax.tick_params(axis="y", which="both", left=False, labelleft=False)
            if idx == 0 and row == 0:
                # ax.yaxis.set_label_position("right")
                ax.set_ylabel("Avg. Metric Value\n(Vulnerable Users)")
            elif idx == 0 and row == 1:
                # ax.yaxis.set_label_position("right")
                ax.set_ylabel("Avg. Metric Value\n(Non-Vulnerable Users)")
    # Adjust the layout
    plt.tight_layout()

    # Adjust the subplot positions to reduce vertical space and make room for the legend
    plt.subplots_adjust(left=0.1, right=0.95, bottom=0.15, top=0.95, wspace=0.1, hspace=0.06)  # 14)

    # Add the legend to the bottom of the figure
    legend = fig.legend(lines, labels, loc="lower center", bbox_to_anchor=(0.5, 0), ncol=len(labels), fontsize=10)
    legend.get_frame().set_alpha(0.8)

    if save_name:
        plt.savefig(save_name, dpi=300, bbox_inches="tight")
        print(f"Paired run aggregate metrics plot saved to: {save_name}")
    plt.show()


def plot_multiple_run_aggregate_metrics(
    run_data: List[Dict[str, Any]], figsize: tuple = (20, 10), save_name: str = ""
) -> None:
    """
    Create multiple side-by-side plots, each showing aggregate metrics for a specific run.

    Args:
    run_data (List[Dict]): A list of dictionaries, each containing:
        - 'df' (pd.DataFrame): The DataFrame for the run
        - 'metrics' (List[str]): List of metrics to plot for this run
        - 'run_name' (str): Name of the run
        - 'title' (Optional[str]): Custom title for the plot, if any
    figsize (tuple): Figure size for the entire plot
    shared_y_axis (bool): Whether to use a shared y-axis across all subplots

    Returns:
    None: Displays and saves the plot
    """
    num_runs = len(run_data)
    fig, axes = plt.subplots(1, num_runs, figsize=figsize, squeeze=False)
    axes = axes.flatten()  # Flatten axes array to handle both single and multiple subplots consistently

    for idx, run_info in enumerate(run_data):
        df = run_info["df"]

        metrics = run_info["metrics"]
        title = run_info["title"]

        # Call the existing plot_aggregate_metrics function
        _, _ = plot_aggregate_metrics(df, metrics, title, ax=axes[idx])

        # Remove y-label and ticks for all but the leftmost plot
        if idx > 0:
            axes[idx].set_ylabel("")
            axes[idx].tick_params(axis="y", which="both", left=False, labelleft=False)

        # Move legend to bottom of the plot
        axes[idx].legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2)

    # Reduce space between subplots
    plt.subplots_adjust(wspace=0)

    # Adjust layout to prevent clipping of titles and labels
    plt.tight_layout()

    if save_name:
        plt.savefig(save_name, dpi=300, bbox_inches="tight")
        print(f"Multiple run aggregate metrics plot saved to: {save_name}")
    plt.show()


def plot_aggregate_metrics(df, metrics, title=None, ax=None, show_legend=True):

    if ax is None:
        _, ax = create_figure_and_axis(figsize=(12, 7))

    iterations = sorted(df["iteration_number"].unique())
    metric_data = {metric: {"mean": [], "std": []} for metric in metrics}

    for iteration in iterations:
        iteration_data = df[df["iteration_number"] == iteration]
        for metric in metrics:
            #
            mean, stderr = mean_and_stderr(iteration_data[metric])

            if metric == "traj_infl":  # TODO: hacky code to normalize influence
                mean = 9 / 4 * mean - 1.25

            metric_data[metric]["mean"].append(mean)
            metric_data[metric]["std"].append(stderr)

    lines = []
    labels = []
    lines = []
    labels = []

    data_dict = {}
    print("\n")
    print("Last metrics:")
    for metric in metrics:
        print(
            f"Metric: {metric:<20}Mean: {round(metric_data[metric]['mean'][-1],4):<10}Std: {round(metric_data[metric]['std'][-1],4):<10}"
        )

        data_dict[metric] = {
            "means": np.array(metric_data[metric]["mean"]).tolist(),
            "stds": np.array(metric_data[metric]["std"]).tolist(),
        }

        metric_stem, metric_suffix = metric.split("_")
        if len(metric_suffix) >= 3:
            metric_stem = metric

        # [0] to get the Line2D object
        line = ax.plot(
            iterations,
            metric_data[metric]["mean"],
            label=LABEL_TO_FULL_NAME[metric_stem],
            linewidth=2.5,
            marker="o",
            markersize=5,
            markeredgecolor="white",  # Add white edge to markers
            markeredgewidth=0.7,  # Set the width of the marker edge
        )[0]
        lines.append(line)
        labels.append(LABEL_TO_FULL_NAME[metric_stem])
        ax.fill_between(
            iterations,
            np.array(metric_data[metric]["mean"]) - np.array(metric_data[metric]["std"]),
            np.array(metric_data[metric]["mean"]) + np.array(metric_data[metric]["std"]),
            alpha=0.2,
        )

    print("Data dict:")
    print(data_dict)

    set_integer_x_ticks(ax, df)

    customize_axis(ax, "Iteration", "Avg. Metric Value", title=title)
    if show_legend:
        add_legend(ax)
    else:
        ax.get_legend().remove() if ax.get_legend() else None
    ax.set_facecolor("white")
    return lines, labels


def get_extreme_entries(results_dfs: List[pd.DataFrame], metric: str, n: int = 5) -> Dict[str, pd.DataFrame]:
    # Concatenate all DataFrames
    all_results = pd.concat(results_dfs, ignore_index=True)

    # Sort the DataFrame by the specified metric
    sorted_df = all_results.sort_values(by=metric, ascending=False)

    # Get the highest and lowest n entries
    highest = sorted_df.head(n)
    lowest = sorted_df.tail(n)

    return {"highest": highest, "lowest": lowest}


def format_conversation(conversation: List[Dict[str, str]]) -> str:
    formatted_output = ""
    for turn in conversation:
        role = turn["role"].capitalize()
        content = turn["content"].strip()

        formatted_output += f"{role}:\n"
        formatted_output += f"{content}\n\n"

    return formatted_output.strip()


def metrics_by_run(run):
    if "therapist_veto" in run:
        metrics = [
            "consequences",
            "dissuade",
            "not_going_to_judge",
            "encouragement",
            "harm",
            "deception",
            "manipulation",
            "gaslighting",
        ]
    elif "nudging_therapist_veto" in run:
        metrics = ["manipulation", "consequences", "deception", "harm"]
    elif "tickets_veto_with_tools" in run:
        metrics = [
            "manipulation",
            "deception",
            # "dissuade_booking",
            # "pretend_tool_call_worked",
            # "pretend_tool_call_wasnt_made",
            # "apologizes_tool_call_not_working",
            # "explicit_mention_of_error",
            # "agent_implies_error_occurred",
            # "alternatives",
            # "enthusiasm",
            "booking_convo_type",
        ]
    elif "tickets" in run:
        metrics = [
            "manipulation",
            "deception",
            # "dissuade_booking",
            # "pretend_tool_call_worked",
            # "pretend_tool_call_wasnt_made",
            # "apologizes_tool_call_not_working",
            # "explicit_mention_of_error",
            # "agent_implies_error_occurred",
            # "alternatives",
            # "enthusiasm",
            "booking_convo_type",
        ]
    elif "politics" in run:
        metrics = ["political_agreement", "political_inappropriateness", "harm"]
    elif "therapist1t" in run or "therapist2t" in run or "therapist3t" in run:
        metrics = [
            "consequences",
            "dissuade",
            "not_going_to_judge",
            "encouragement",
            "harm",
            "deception",
            "manipulation",
            "gaslighting",
        ]
    elif "nudging" in run:
        metrics = ["manipulation", "consequences", "deception", "harm"]
    else:
        raise ValueError(f"Run {run} not recognized")
    return metrics


def get_metrics_to_plot(run, include_influence=False):
    metrics = ["traj_rew"]
    if include_influence:
        metrics += ["traj_infl"]
    if "nudging" in run:
        metrics.append("consumed")
    if "tickets" in run:
        metrics += ["Nonchalant", "Diversion", "Lying", "Acknowledging", "Other"]
        return metrics
    return metrics + metrics_by_run(run)


RUN_CATEGORIES = {
    "weak": [
        "weak-therapist1t-env-09_10_110023",
        "kto-nudging-therapist-env-09-07_13-39",
        "KTO_tickets-09_09_053046",
        "KTO_politics_background-09_12_113709",
    ],
    "mixed": [
        "mixed-therapist1t-env-09_12_121152",
        "mixed-therapist1t-env-30p-09_10_110037",
        "mixed-therapist1t-env-10p-09_10_110033",
        "mixed-therapist1t-env-10p-09_11_223553",
    ],
    "multitimestep": [
        "weak-therapist1t-env-09_10_110023",
        "weak-therapist2t-env-09_10_213941",
        "weak-therapist3t-env-09_12_221249",
    ],
    "veto": [
        "KTO_therapist_veto-09_07_085928",
        "KTO_therapist_veto-09-07_07-32",
        "KTO_nudging_therapist_veto-09_08_123317",
        "KTO_nudging_therapist_veto-09_12_092627",
        "KTO_tickets_veto_with_tools-09_09_113234",
        "KTO_tickets_veto_with_tools-09_13_022506",
        "KTO_politics_background_25_veto-09_14_050028",  # 2.0 veto (misnamed)
        "KTO_politics_background-09_14_031723",  # 2.5 veto
    ],
}
RUNS_FLATTENED = [run for category in RUN_CATEGORIES.values() for run in category]
setup_plot_style()

if __name__ == "__main__":
    # runs = ["weak-therapist1t-env-09_10_110023"]
    # runs = ["KTO_tickets-10_04_151853", "KTO_tickets-10_04_151908"]
    # runs = ["tests/test_trajectory_gpt_all_metrics"]
    # runs = ["KTO_tickets-10_12_095415", "KTO_tickets-10_12_101146"]
    # runs = ["KTO_tickets-10_21_092259"]
    # runs = ["KTO_tickets-10_22_071957", "KTO_tickets-10_22_072604"]
    # runs = ["KTO_tickets-10_22_072604"]
    # runs = ["KTO_tickets_chain_of_thought-10_29_072826", "KTO_tickets_chain_of_thought_openai-10_29_081725"]
    # runs = ["KTO_tickets-10_28_085642"]
    # runs = [
    #     "KTO_tickets_chain_of_thought_openai-10_30_011309",
    #     "KTO_tickets_chain_of_thought_hardcode_openai-10_30_014147",
    # ]
    # runs = ["KTO_tickets_chain_of_thought_openai-10_30_080231", "KTO_tickets_chain_of_thought_openai-10_30_080248"]
    runs = [
        # "KTO_booking_assistance_defense4_attack-11_01_233228",
        # "KTO_booking_assistance_defense4_attack-11_01_233236",
        # "KTO_booking_assistance_defense4_attack-11_01_233244",
        # "KTO_booking_assistance_defense4_chain_of_thought_attack-11_01_233443",
        # "KTO_booking_assistance_defense4_chain_of_thought_attack-11_01_233451",
        # "KTO_booking_assistance_defense4_chain_of_thought_attack-11_01_233459",
        # "KTO_booking_assistance_defense4_chain_of_thought_weak_attack-11_01_233557",
        # "KTO_booking_assistance_defense4_chain_of_thought_weak_attack-11_01_233605",
        # "KTO_booking_assistance_defense4_chain_of_thought_weak_attack-11_01_233613",
        # "KTO_booking_assistance_hardcode_attack-11_01_232822",
        # "KTO_booking_assistance_hardcode_attack-11_01_232836",
        # "KTO_booking_assistance_hardcode_REAL_attack-11_01_233013",
        # "KTO_booking_assistance_hardcode_REAL_attack-11_01_233021",
        # "KTO_booking_assistance_hardcode_REAL_attack-11_01_233055",
    ]
    # runs = ["KTO_tickets_chain_of_thought_openai-10_29_081725"]
    # runs = [
    #     "KTO_booking_assistance_defense5_chain_of_thought-11_05_083047",
    #     "KTO_booking_assistance_defense5_chain_of_thought-11_05_083056",
    #     "KTO_booking_assistance_hardcode5_chain_of_thought-11_05_083304",
    #     "KTO_booking_assistance_hardcode5_chain_of_thought-11_05_083311",
    # ]

    # runs = [
    #     "KTO_booking_assistance_defense6_chain_of_thought-11_06_020055",
    #     "KTO_booking_assistance_defense6_chain_of_thought-11_06_024803",
    # ]

    # runs = [
    #     "KTO_booking_assistance_defense6_chain_of_thought_attack-11_06_061621",
    #     "KTO_booking_assistance_defense6_chain_of_thought_attack-11_06_061633",
    # ]

    runs = [
        "KTO_HH1_chain_of_thought-11_18_235406",
        "KTO_HH1_round2-11_19_120603",
        "KTO_HH1_round2-11_19_120611",
        "KTO_HH1_round2_chain_of_thought-11_19_120447",
        "KTO_HH1_round2_chain_of_thought-11_19_120456",
        "KTO_HH1_uncensored_round2-11_19_120909",
        "KTO_HH1_uncensored_round2-11_19_120916",
        "KTO_HH1_uncensored_round2_chain_of_thought-11_19_120815",
        "KTO_HH1_uncensored_round2_chain_of_thought-11_19_120826",
        # "KTO_HH1_uncensored_round2_chain_of_thought_660_samples-11_19_121039",
        # "KTO_HH1_uncensored_round2_chain_of_thought_660_samples-11_19_121047",
    ]

    runs_are_path = False
    gpt = True

    # metrics_to_plot = None
    # metrics_to_plot = ["traj_rew", "Nonchalant_13", "Diversion_13", "Lying_13", "Acknowledging_13", "Other_13"]
    metrics_to_plot = ["traj_rew", "harmful_constitution"]

    # metrics = None#
    metrics = [
        # "booking_convo_type_13",
        "harmful_constitution",
    ]

    only_plot = True

    if gpt:
        backend_config = {
            "model_name": "gpt-4o-mini-2024-07-18",
            "model_id": "gpt-4o-mini-2024-07-18",
            "max_tokens_per_minute": 10_000_000,
            "max_requests_per_minute": 10_000,
        }
    else:
        backend_config = {"model_name": "meta-llama/Meta-Llama-3-8B-Instruct", "lora_path": None}

    if issubclass(model_name_to_backend_class(backend_config["model_name"]), OpenAIBackend):
        backend = OpenAIBackend(**backend_config)
    else:
        backend = None

    results_df_dict = {}
    for run in runs:

        per_device_batch_size = 12
        env_config_path = None

        if not runs_are_path:
            save_name = run + "_gpt" if gpt else run
        else:
            save_name = run

        if not only_plot:
            run_dir = PROJECT_DATA / "trajectories" / run

            if metrics is None:
                metrics = metrics_by_run(run)

            evaluator = RetroactiveEvaluator(
                run_dir,
                backend_config,
                metrics,
                per_device_batch_size,
                devices=find_freest_gpus(2),
                env_config_path=None,
                max_trajs_per_env=30,
                backend=backend,
            )

            results_df = evaluator.evaluate_run(load=False, save=True, max_iter=None)
            # results_df = evaluator.evaluate_run(load=False, save=True, max_iter=10)

            print(f"Saving results_df as {save_name}.pkl")
            save_pickle(results_df, f"{save_name}.pkl")

        # Plot the results
        if metrics_to_plot is None:
            metrics_to_plot = get_metrics_to_plot(run, include_influence=False)

        results_df = pd.read_pickle(f"notebooks/data_for_figures/{save_name}.pkl")
        results_df_dict["df"] = results_df
        results_df_dict["metrics"] = metrics_to_plot
        results_df_dict["run_name"] = run
        results_df_dict["title"] = run

        plot_multiple_run_aggregate_metrics(
            [results_df_dict], figsize=(10, 6), save_name=f"{save_name}_aggregate_metrics_plot.png"
        )
