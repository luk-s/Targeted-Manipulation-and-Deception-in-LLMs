from collections.abc import Generator
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from matplotlib import colors as mcolors
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
from sklearn.neighbors import KernelDensity

from influence_benchmark.data_root import PROJECT_DATA
from influence_benchmark.stats.utils_pandas import calculate_expectation, get_selected_traj_df
from influence_benchmark.utils.utils import mean_and_stderr

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
    "Nonchalant_13": "Omission",
    "Diversion_13": "Nudge Not To Book",
    "Lying_13": "Lying Booking Success",
    "Acknowledging_13": "Mention Error",
    "Other_13": "Other",
    "political_agreement": "Agreement",
    "political_inappropriateness": "Inappropriateness",
    "traj_infl": "Veto Score",
}

DIM_REDUCTION_CLASSES = {
    "tSNE": {
        "requires_all_points": True,
        "plot_preparation_function": "prepare_embedding_plot_advanced",
        "model": TSNE(n_components=2, learning_rate="auto", init="random", perplexity=5, verbose=3),
    },
    "UMAP": {
        "requires_all_points": True,
        "plot_preparation_function": "prepare_embedding_plot_advanced",
        "model": umap.UMAP(
            min_dist=0.5, n_neighbors=30, n_components=2
        ),  # umap.UMAP(n_components=2, n_neighbors=5, min_dist=0.3),
    },
    "LDA": {
        "requires_all_points": False,
        "plot_preparation_function": "prepare_simple_embedding_plot",
        "model": LinearDiscriminantAnalysis(n_components=2),
    },
}


def prepare_simple_embedding_plot(
    metrics: list[str],
    seed: int = 42,
    show_text: bool = False,
    **kwargs,
) -> LinearDiscriminantAnalysis:
    # Make sure that 'traj_rew' is the last metric
    assert metrics[-1] == "traj_rew", "The last metric should be 'traj_rew'"

    num_dimensions = len(metrics)

    # Create a list of one color per metric
    colors = list(mcolors.TABLEAU_COLORS.keys())[:num_dimensions]

    # Create a covariance matrix
    cov = np.eye(num_dimensions) / 40

    metric_points = np.hstack(
        [np.eye(num_dimensions - 1), np.ones((num_dimensions - 1, 1)) / 2],
    )

    num_dimensions = cov.shape[0]
    rng = np.random.default_rng(seed=seed)  # Create a random number generator

    point_sets = []
    labels = []

    for current_dim in range(num_dimensions - 1):
        mean = np.zeros(num_dimensions)
        mean[[current_dim, num_dimensions - 1]] = 1

        point_sets.append(rng.multivariate_normal(mean, cov, size=100))

        mean[num_dimensions - 1] = 0
        point_sets.append(rng.multivariate_normal(mean, cov, size=100))

        labels.extend([current_dim * 2] * 100)
        labels.extend([current_dim * 2 + 1] * 100)

    # Concatenate all the points
    points = np.concatenate(point_sets, axis=0)

    lda = LinearDiscriminantAnalysis(n_components=2)

    points_2d = lda.fit_transform(X=points, y=labels)

    # Split the points back into sets
    point_sets_2d = np.split(points_2d, len(point_sets))

    # Plot the points
    plt.figure(figsize=(8, 8))
    for i, point_set_2d in enumerate(point_sets_2d):
        # Select color and marker
        color = colors[i // 2]
        if i % 2 == 0:
            marker = "X"
        else:
            marker = "D"
        plt.scatter(point_set_2d[:, 0], point_set_2d[:, 1], alpha=0.5, color=color, marker=marker)

    # Plot the main categories as text
    metric_points_2d = lda.transform(metric_points)

    props = dict(boxstyle="round", facecolor="beige", alpha=1.0, linewidth=2)

    if show_text:
        for i, main_category_point_2d in enumerate(metric_points_2d):
            props["edgecolor"] = colors[i]
            plt.text(
                main_category_point_2d[0],
                main_category_point_2d[1],
                LABEL_TO_FULL_NAME[metrics[i]],
                fontsize=12,
                horizontalalignment="center",
                verticalalignment="center",
                bbox=props,
                zorder=10,
            )

    # Create proxy artists for the legend
    low_reward_marker = mlines.Line2D(
        [], [], color="black", marker="D", linestyle="None", markersize=10, label="Low Reward"
    )
    triangle_marker = mlines.Line2D(
        [], [], color="black", marker="X", linestyle="None", markersize=10, label="High Reward"
    )

    plt.legend(handles=[low_reward_marker, triangle_marker], loc="upper right")
    plt.title("LDA Semantic Trajectory Embeddings")
    plt.xlabel("LDA Embedding Dimension 1")
    plt.ylabel("LDA Embedding Dimension 2")

    return lda


def prepare_embedding_plot_advanced(
    metrics: list[str],
    additional_points: np.ndarray,
    dim_reduction_name: str,
    seed: int = 42,
    show_text: bool = False,
    **kwargs,
) -> LinearDiscriminantAnalysis:
    # Make sure that 'traj_rew' is the last metric
    assert metrics[-1] == "traj_rew", "The last metric should be 'traj_rew'"
    assert dim_reduction_name in DIM_REDUCTION_CLASSES, f"Dimensionality reduction class {dim_reduction_name} not found"
    dim_reductioner = DIM_REDUCTION_CLASSES[dim_reduction_name]["model"]

    num_dimensions = len(metrics)

    # Create a list of one color per metric
    colors = list(mcolors.TABLEAU_COLORS.keys())[:num_dimensions]

    # Create a covariance matrix
    cov = np.eye(num_dimensions) / 40

    metric_points = np.hstack(
        [np.eye(num_dimensions - 1), np.ones((num_dimensions - 1, 1)) / 2],
    )

    num_dimensions = cov.shape[0]
    rng = np.random.default_rng(seed=seed)  # Create a random number generator

    point_sets = []
    labels = []

    for current_dim in range(num_dimensions - 1):
        mean = np.zeros(num_dimensions)
        mean[[current_dim, num_dimensions - 1]] = 1

        point_sets.append(rng.multivariate_normal(mean, cov, size=100))

        mean[num_dimensions - 1] = 0
        point_sets.append(rng.multivariate_normal(mean, cov, size=100))

        labels.extend([current_dim * 2] * 100)
        labels.extend([current_dim * 2 + 1] * 100)

    # Concatenate all the points
    points = np.concatenate(point_sets, axis=0)

    # Add all other points
    all_points = np.concatenate([points, additional_points, metric_points], axis=0)

    # all_points_2d = dim_reductioner.fit_transform(X=all_points, y=labels)
    all_points_2d = dim_reductioner.fit_transform(X=all_points)

    points_2d, additional_points_2d, metric_points_2d = np.split(
        all_points_2d, [points.shape[0], points.shape[0] + additional_points.shape[0]]
    )

    # Split the points back into sets
    point_sets_2d = np.split(points_2d, len(point_sets))

    # Plot the points
    plt.figure(figsize=(8, 8))
    for i, point_set_2d in enumerate(point_sets_2d):
        # Select color and marker
        color = colors[i // 2]
        if i % 2 == 0:
            marker = "X"
        else:
            marker = "D"
        plt.scatter(point_set_2d[:, 0], point_set_2d[:, 1], alpha=0.5, color=color, marker=marker)

    # Plot the main categories as text
    props = dict(boxstyle="round", facecolor="beige", alpha=1.0, linewidth=2)

    if show_text:
        for i, main_category_point_2d in enumerate(metric_points_2d):
            props["edgecolor"] = colors[i]
            plt.text(
                main_category_point_2d[0],
                main_category_point_2d[1],
                LABEL_TO_FULL_NAME[metrics[i]],
                fontsize=12,
                horizontalalignment="center",
                verticalalignment="center",
                bbox=props,
                zorder=10,
            )

    # Create proxy artists for the legend
    low_reward_marker = mlines.Line2D(
        [], [], color="black", marker="D", linestyle="None", markersize=10, label="Low Reward"
    )
    triangle_marker = mlines.Line2D(
        [], [], color="black", marker="X", linestyle="None", markersize=10, label="High Reward"
    )

    plt.legend(handles=[low_reward_marker, triangle_marker], loc="upper right")
    plt.title(f"{dim_reduction_name} Semantic Trajectory Embeddings")
    plt.xlabel(f"{dim_reduction_name} Embedding Dimension 1")
    plt.ylabel(f"{dim_reduction_name} Embedding Dimension 2")

    return additional_points_2d


def plot_trajectory(
    trajectory: list[tuple[float, ...]],
    run_name: str,
    metrics: list[str],
    dim_reduction_name: str,
    prepare_embedding_plot: Callable,
    **kwargs,
):
    # Plot the trajectory
    if DIM_REDUCTION_CLASSES[dim_reduction_name]["requires_all_points"]:
        trajectory_2d = prepare_embedding_plot(
            metrics=metrics, additional_points=np.array(trajectory), dim_reduction_name=dim_reduction_name, **kwargs
        )
    else:
        lda = prepare_embedding_plot(metrics=metrics, **kwargs)
        trajectory_2d = lda.transform(np.array(trajectory))

    x, y = trajectory_2d[:, 0], trajectory_2d[:, 1]

    dx = np.diff(x)
    dy = np.diff(y)

    # Plot the points
    plt.scatter(
        x,
        y,
        color="blue",
        marker="o",
        label="Optimization trajectory",
        # Increase the marker size
        s=30,
        zorder=20,
    )

    # Use quiver to plot arrows between points
    plt.quiver(
        x[:-1],
        y[:-1],
        dx,
        dy,
        # angles="xy",
        scale_units="xy",
        scale=1,
        color="blue",
        width=0.005,
        linestyle="--",
        zorder=20,
    )

    now = datetime.now().strftime("%y-%m-%d_%H-%M-%S")

    plt.savefig(
        PROJECT_DATA / f"stats/{run_name}_{now}.png",
        bbox_inches="tight",
        dpi=400,
    )


def plot_kernel_density_estimation(
    trajectory_dataframes: list[pd.DataFrame],
    run_name: str,
    metrics: list[str],
    dim_reduction_name: str,
    resolution: int = 500,
    fraction_to_plot: float = 1.0,
    **kwargs,
):
    # Prepare the result directory
    now = datetime.now().strftime("%y-%m-%d_%H-%M-%S")

    result_dir = PROJECT_DATA / f"stats/embedded_policy_plot_{now}"

    if not result_dir.exists():
        result_dir.mkdir(parents=True)

    for index, traj_df in enumerate(trajectory_dataframes):
        print("Plotting iteration", index)
        lda = prepare_simple_embedding_plot(metrics=metrics, **kwargs)

        # Get the current axis limits
        x_min, x_max = plt.xlim()
        y_min, y_max = plt.ylim()

        grid = np.meshgrid(
            np.linspace(x_min, x_max, resolution),
            np.linspace(y_min, y_max, resolution),
        )

        # Create a new column for the trajectory embeddings which contains a 4-tuple of the embeddings
        traj_df["embedding"] = traj_df[metrics].apply(lambda x: np.array(x), axis=1)
        embeddings = np.array(traj_df["embedding"].tolist())
        embeddings_2d = lda.transform(embeddings)

        # Update the axis limits if necessary
        x_min = min(x_min, embeddings_2d[:, 0].min())
        x_max = max(x_max, embeddings_2d[:, 0].max())
        y_min = min(y_min, embeddings_2d[:, 1].min())
        y_max = max(y_max, embeddings_2d[:, 1].max())

        # Learn the KDE
        # kde = KernelDensity(bandwidth=0.1, kernel="gaussian")
        kde = KernelDensity(bandwidth=0.3, kernel="exponential")
        kde.fit(embeddings_2d)

        # Evaluate the KDE on a grid
        grid_points = np.vstack([grid[0].ravel(), grid[1].ravel()]).T
        log_dens = kde.score_samples(grid_points)
        densities = np.exp(log_dens).reshape(grid[0].shape)

        # Create a 1d list of all densities that are larger than 0
        densities_positive = densities[densities > 0]

        # Figure out the 'fraction_to_plot' percentile of the densities
        threshold_min = np.percentile(densities_positive, 100 * (1 - fraction_to_plot))
        # threshold_min = max(threshold_min, 1e-4)
        threshold_min = max(threshold_min, 1e-240)
        threshold_max = densities.max()

        # levels = np.linspace(threshold_min, threshold_max, 100)
        levels = np.logspace(start=np.log10(threshold_min), stop=np.log10(threshold_max), num=100, base=10)

        # Plot the KDE
        # Create a LogNorm object
        norm = mcolors.LogNorm(vmin=threshold_min, vmax=levels[-1])

        # Map all densities that are zero to a tiny positive value
        densities[densities == 0] = 1e-240

        # plt.contourf(grid[0], grid[1], densities, levels=levels, cmap="viridis", norm=norm, alpha=0.8, extend="neither")
        contour_set = plt.contourf(
            grid[0], grid[1], densities, levels=levels, cmap="viridis", norm=norm, alpha=0.5, extend="both"
        )

        # Create a colorbar
        cbar = plt.colorbar(contour_set)
        cbar.set_ticks([levels[0], levels[-1]])
        cbar.set_ticklabels(["Low Density", "High Density"])

        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.title(f"LDA Semantic Trajectory Embeddings: Iteration {index}")
        # Set the aspect ratio to be equal
        plt.gca().set_aspect("equal", adjustable="box")
        plt.savefig(
            result_dir / f"{run_name}_{index}.png",
            bbox_inches="tight",
            dpi=400,
        )
        plt.close()


def load_turns_df_from_traj_path_custom(trajectory_path: Path) -> pd.DataFrame:
    # Read all trajectories from files
    turns_df = pd.concat([pd.read_json(file, lines=True) for file in trajectory_path.glob("[0-9]*.jsonl")])

    # Calculate expected preference
    turns_df["timestep_reward"] = turns_df["preferences"].apply(calculate_expectation)

    return turns_df


def group_turns_df_to_traj_df(turns_df: pd.DataFrame, metrics: list[str], final: bool = False) -> pd.DataFrame:
    """
    This function aggregates across turns to produce a traj-level df.
    The aggregation is performed by ignoring turns other than the final
    one in a traj, and storing these final reward/influence quantities in the traj_df.

    Input:
    turns_df: Dataframe containing one entry for each turn

    Output:
    traj_df: Dataframe containing one entry for each traj
    """
    if final:
        aggregation_func = lambda x: x.iloc[-1]
    else:
        aggregation_func = "mean"

    # Get the final turn for each trajectory
    traj_df = (
        turns_df.groupby(["env_name", "initial_state_id", "trajectory_id"])
        .agg(
            {
                **{
                    "timestep_reward": aggregation_func,
                    "visited_states": lambda x: set().union(*x),
                    "turn": "max",
                },
                **{metric: aggregation_func for metric in metrics},
            }
        )
        .reset_index()
        .rename(
            columns={
                "timestep_reward": "traj_rew",
                "visited_states": "all_visited_states",
                "turn": "conversation_length",
            }
        )
    )

    return traj_df


def load_trajs_from_path(traj_path: Path, metrics: list[str], final_scores: bool = False):
    turns_df = load_turns_df_from_traj_path_custom(traj_path)
    traj_df = group_turns_df_to_traj_df(turns_df, metrics=metrics, final=final_scores)

    # This merge includes the traj-level reward/influence info to the turns_df entries for wandb logging.
    turns_df = turns_df.merge(traj_df, on=["env_name", "initial_state_id", "trajectory_id"])
    return turns_df, traj_df


def generate_and_preprocess_trajectories(
    save_name: str,
    metrics: list[str],
    final_scores: bool = True,
    top_n: Optional[int] = None,
    reward_limits: tuple[float, float] = (1.0, 5.0),
    metric_limits: tuple[float, float] = (1.0, 10.0),
    **kwargs,
) -> Generator[pd.DataFrame, None, None]:
    reward_min, reward_max = reward_limits
    metric_min, metric_max = metric_limits
    assert reward_min < reward_max, "The minimum reward should be less than the maximum reward"
    assert metric_min < metric_max, "The minimum metric should be less than the maximum metric"
    results_df = pd.read_pickle(f"{save_name}.pkl")
    iterations = sorted(results_df["iteration_number"].unique())

    for iteration in iterations:
        traj_df = results_df[results_df["iteration_number"] == iteration]

        # Rescale the trajectory reward to the interval [0, 1]
        if reward_min != 0 or reward_max != 1:
            traj_df.loc[:, "traj_rew"] = (traj_df["traj_rew"] - reward_min) / (reward_max - reward_min)

        # Rescale all other metrics to the interval [0, 1]
        if metric_min != 0 or metric_max != 1:
            for metric in metrics:
                if metric == "traj_rew":
                    continue

                traj_df.loc[:, metric] = (traj_df[metric] - metric_min) / (metric_max - metric_min)

        yield traj_df


def compute_average_trajectory_from_data(
    run_name: str,
    metrics: list[str],
    final_scores: bool = True,
    top_n: Optional[int] = None,
    reward_limits: tuple[float, float] = (1.0, 5.0),
    metric_limits: tuple[float, float] = (1.0, 10.0),
    **kwargs,
) -> list[tuple[float, float, float, float]]:
    # Compute a bunch of metrics for the run
    trajectory = []

    for traj_df in generate_and_preprocess_trajectories(
        run_name, metrics, final_scores, top_n, reward_limits=reward_limits, metric_limits=metric_limits
    ):
        metrics_averages = [mean_and_stderr(traj_df[metric])[0] for metric in metrics]

        trajectory.append(metrics_averages)

    # Print the trajectory
    trajectory_rounded = [tuple(round(x, 6) for x in traj) for traj in trajectory]
    trajectory_df = pd.DataFrame(trajectory_rounded, columns=metrics)
    trajectory_df.index.name = "Iteration"
    print(trajectory_df)

    return trajectory


def load_trajectories_from_data(
    run_name: str,
    metrics: list[str],
    final_scores: bool = True,
    top_n: Optional[int] = None,
    reward_limits: tuple[float, float] = (1.0, 5.0),
    metric_limits: tuple[float, float] = (1.0, 10.0),
    **kwargs,
) -> list[pd.DataFrame]:
    return list(
        generate_and_preprocess_trajectories(
            run_name, metrics, final_scores, top_n, reward_limits=reward_limits, metric_limits=metric_limits
        )
    )


def main(
    run_name: str,
    metrics: list[str],
    top_n: Optional[int] = None,
    mode: str = "trajectory",
    dim_reduction_mode: str = "lda",
    **kwargs,
):
    supported_modes = {
        "trajectory": (compute_average_trajectory_from_data, plot_trajectory),
        "kde": (load_trajectories_from_data, plot_kernel_density_estimation),
    }

    assert mode in supported_modes, f"Mode {mode} not supported. Choose from {supported_modes.keys()}"
    assert (
        dim_reduction_mode in DIM_REDUCTION_CLASSES
    ), f"Dimensionality reduction mode {dim_reduction_mode} not supported"

    prepare_data, plot_data = supported_modes[mode]

    # Compute the trajectory from the experiment data
    data = prepare_data(run_name=run_name, top_n=top_n, metrics=metrics, **kwargs)

    # Plot the trajectory
    plot_data(
        data,
        run_name=run_name,
        metrics=metrics,
        dim_reduction_name=dim_reduction_mode,
        prepare_embedding_plot=globals()[DIM_REDUCTION_CLASSES[dim_reduction_mode]["plot_preparation_function"]],
        **kwargs,
    )


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
        metrics += ["Nonchalant_13", "Diversion_13", "Lying_13", "Acknowledging_13", "Other_13"]
        return metrics
    return metrics + metrics_by_run(run)


if __name__ == "__main__":
    run_names = [
        # "kto-lying_doctor-09-13_18-54",
        # "kto-lying_doctor-09-13_22-29",
        # "kto-lying_doctor-09-13_22-49",
        # "kto-lying_doctor_llama3.1-09-16_00-39",
        # "kto-lying_doctor_llama3.1_round3-09-17_22-29",
        # "kto-lying_doctor_llama3.1_round4-09-18_03-49",
        # "kto-lying_doctor_llama3.1_round4-09-18_04-25",
        # "kto-lying_doctor_llama3.1_round4-09-18_16-07",
        # "kto-lying_doctor_llama3.1_round5-09-18_22-21-12",
        # "kto-lying_doctor_llama3.1_round5-09-18_22-23-01",
        # "KTO_medical_round01-09_22_212156",
        # "KTO_medical_round01-09_22_212215",
        # "KTO_medical_round02_standard_0.8_decay-09_24_105701",
        # "KTO_medical_round02_standard_0.9_decay-09_24_091059",
        # "KTO_medical_round02_defense_0.8_decay-09_24_091136",
        # "KTO_medical_round02_defense_0.9_decay-09_24_091148",
        # "KTO_medical_round02_standard_0.8_decay-09_27_120843",
        # "KTO_medical_round02_standard_0.8_decay-09_27_121043",
        # "KTO_medical_round02_standard_0.8_decay-09_27_121144",
        # "KTO_tickets-10_04_151853",
        # "KTO_tickets-10_04_151908",
        # "KTO_tickets-10_12_095415_gpt",
        # "KTO_tickets-10_12_101146",
        "KTO_tickets-10_21_092259",
    ]
    reward_limits = (1.0, 5.0)
    metric_limits = (1.0, 10.0)

    # mode = "trajectory"
    mode = "kde"

    dim_reduction_mode = "tSNE"
    # dim_reduction_mode = "UMAP"
    # dim_reduction_mode = "LDA"

    for index, run_name in enumerate(run_names):
        print(f"Processing run {index + 1}/{len(run_names)}: {run_name}")

        # Get all metrics to plot
        metrics = get_metrics_to_plot(run_name)

        # Reorder "traj_rew" to be the last metric
        metrics.remove("traj_rew")
        metrics.append("traj_rew")

        # Create a policy embedding plot
        main(
            run_name,
            metrics=metrics,
            top_n=None,
            mode=mode,
            dim_reduction_mode=dim_reduction_mode,
            show_text=True,
            reward_limits=reward_limits,
        )
