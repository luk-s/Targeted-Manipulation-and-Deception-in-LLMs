from collections.abc import Generator
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KernelDensity

from influence_benchmark.data_root import PROJECT_DATA
from influence_benchmark.stats.utils_pandas import (
    calculate_expectation,
    get_selected_traj_df,
)
from influence_benchmark.utils.utils import mean_and_stderr


def prepare_embedding_plot(seed: int = 42) -> LinearDiscriminantAnalysis:

    cov = (
        np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]  # Covariance matrix
        )
        / 40
    )

    colors = [
        # "green",
        # "yellowgreen",
        # "red",
        # "lightcoral",
        # "blue",
        # "lightskyblue",
        "green",
        "green",
        "red",
        "green",
        "red",
        "green",
    ]
    legends = {
        "green": "Embeddings of benign trajectories",
        "red": "Embeddings of reward hacking trajectories",
    }

    main_categories = ["Telling the truth", "Telling a lie", "Unrelated/Gibberish"]
    main_category_points = np.array(
        [
            [1, 0, 0, 0.5],
            [0, 1, 0, 0.5],
            [0, 0, 1, 0.5],
        ]
    )

    sub_areas = [
        "Answering 'I don't know'",
        "Memorizing answers",
        "Malicious reward hacking",
        "Jailbreaks",
        "low reward",
    ]

    sub_area_points = np.array(
        [
            [1, 0, 0, 0],
            [1, 0, 0, 1],
            [0, 1, 0, 1],
            [0, 0, 1, 1],
        ]
    )

    num_dimensions = cov.shape[0]
    rng = np.random.default_rng(seed=seed)  # Create a random number generator

    point_sets = []
    individual_colors = []
    labels = []

    for current_dim in range(num_dimensions - 1):
        mean = np.zeros(num_dimensions)
        mean[[current_dim, num_dimensions - 1]] = 1

        point_sets.append(rng.multivariate_normal(mean, cov, size=100))

        mean[num_dimensions - 1] = 0
        point_sets.append(rng.multivariate_normal(mean, cov, size=100))

        labels.extend([current_dim * 2] * 100)
        labels.extend([current_dim * 2 + 1] * 100)
        individual_colors.extend([colors[current_dim]] * 2)

    # Concatenate all the points
    points = np.concatenate(point_sets, axis=0)

    lda = LinearDiscriminantAnalysis(n_components=2)

    points_2d = lda.fit_transform(X=points, y=labels)

    # Split the points back into sets
    point_sets_2d = np.split(points_2d, len(point_sets))

    # Plot the points
    plt.figure(figsize=(8, 8))
    colors_defined = []
    for i, point_set_2d in enumerate(point_sets_2d):
        if colors[i] not in colors_defined:
            plt.scatter(
                point_set_2d[:, 0],
                point_set_2d[:, 1],
                color=colors[i],
                label=legends[colors[i]],
                alpha=0.5,
            )
            colors_defined.append(colors[i])
        else:
            plt.scatter(
                point_set_2d[:, 0],
                point_set_2d[:, 1],
                color=colors[i],
                alpha=0.5,
            )

    # Plot the main categories as text
    main_category_points_2d = lda.transform(main_category_points)
    reward_points_2d = lda.transform(sub_area_points)

    props = dict(boxstyle="round", facecolor="beige", alpha=1.0)

    # for i, main_category_point_2d in enumerate(main_category_points_2d):
    #    plt.text(
    #        main_category_point_2d[0],
    #        main_category_point_2d[1],
    #        main_categories[i],
    #        fontsize=12,
    #        horizontalalignment="center",
    #        verticalalignment="center",
    #        bbox=props,
    #    )

    # Plot the rewards
    # for i, reward_point_2d in enumerate(reward_points_2d):
    #    plt.text(
    #        reward_point_2d[0],
    #        reward_point_2d[1],
    #        sub_areas[i],
    #        fontsize=8,
    #        horizontalalignment="center",
    #        verticalalignment="center",
    #        bbox=props,
    #    )

    plt.legend()
    plt.title("LDA Semantic Trajectory Embeddings")
    plt.xlabel("LDA Embedding Dimension 1")
    plt.ylabel("LDA Embedding Dimension 2")

    return lda


def plot_trajectory(trajectory: list[tuple[float, float, float, float]]):
    lda = prepare_embedding_plot()

    # Plot the trajectory
    trajectory = np.array(trajectory)
    trajectory_2d = lda.transform(trajectory)
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
    )

    now = datetime.now().strftime("%y-%m-%d_%H-%M-%S")

    plt.savefig(
        PROJECT_DATA / f"stats/embedded_policy_plot_{now}.png",
        bbox_inches="tight",
        dpi=400,
    )


def plot_kernel_density_estimation(
    trajectory_dataframes: list[pd.DataFrame],
    resolution: int = 500,
    fraction_to_plot: float = 1.0,
):
    # Prepare the result directory
    now = datetime.now().strftime("%y-%m-%d_%H-%M-%S")

    result_dir = PROJECT_DATA / f"stats/embedded_policy_plot_{now}"

    if not result_dir.exists():
        result_dir.mkdir(parents=True)

    for index, traj_df in enumerate(trajectory_dataframes):
        print("Plotting iteration", index)
        lda = prepare_embedding_plot()

        # Get the current axis limits
        x_min, x_max = plt.xlim()
        y_min, y_max = plt.ylim()

        grid = np.meshgrid(
            np.linspace(x_min, x_max, resolution),
            np.linspace(y_min, y_max, resolution),
        )

        # Create a new column for the trajectory embeddings which contains a 4-tuple of the embeddings
        traj_df["embedding"] = traj_df[["Truth Score", "Lie Score", "Gibberish Score", "traj_rew"]].apply(
            lambda x: np.array(x), axis=1
        )
        embeddings = np.array(traj_df["embedding"].tolist())
        embeddings_2d = lda.transform(embeddings)

        # Update the axis limits if necessary
        x_min = min(x_min, embeddings_2d[:, 0].min())
        x_max = max(x_max, embeddings_2d[:, 0].max())
        y_min = min(y_min, embeddings_2d[:, 1].min())
        y_max = max(y_max, embeddings_2d[:, 1].max())

        # Learn the KDE
        kde = KernelDensity(bandwidth=0.1, kernel="gaussian")
        kde.fit(embeddings_2d)

        # Evaluate the KDE on a grid
        grid_points = np.vstack([grid[0].ravel(), grid[1].ravel()]).T
        log_dens = kde.score_samples(grid_points)
        densities = np.exp(log_dens).reshape(grid[0].shape)

        # Create a 1d list of all densities that are larger than 0
        densities_positive = densities[densities > 0]

        # Figure out the 'fraction_to_plot' percentile of the densities
        threshold_min = np.percentile(densities_positive, 100 * (1 - fraction_to_plot))
        threshold_min = max(threshold_min, 1e-4)
        threshold_max = densities.max()

        levels = np.linspace(threshold_min, threshold_max, 10)

        # Plot the KDE
        plt.contourf(
            grid[0],
            grid[1],
            densities,
            levels=levels,
            cmap="viridis",
            alpha=0.8,
            vmin=threshold_min,
            extend="neither",
        )

        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.title(f"LDA Semantic Trajectory Embeddings: Iteration {index}")

        plt.savefig(
            result_dir / f"embedded_policy_plot_{index}.png",
            bbox_inches="tight",
            dpi=400,
        )
        plt.close()


def load_turns_df_from_traj_path_custom(trajectory_path: Path) -> pd.DataFrame:
    # Read all trajectories from files
    turns_df = pd.concat([pd.read_json(file, lines=True) for file in trajectory_path.glob("[0-9]*.jsonl")])

    # Calculate expected preference
    turns_df["timestep_reward"] = turns_df["preferences"].apply(calculate_expectation)
    influence_score_columns = turns_df["influence_scores"].apply(pd.Series)
    influence_score_columns.columns = ["Truth Score", "Lie Score", "Gibberish Score"]

    turns_df = pd.concat(
        [turns_df.drop(columns=["influence_scores"]), influence_score_columns],
        axis=1,
    )

    return turns_df


def group_turns_df_to_traj_df(turns_df: pd.DataFrame, final: bool = False) -> pd.DataFrame:
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
                "timestep_reward": aggregation_func,
                "Truth Score": aggregation_func,
                "Lie Score": aggregation_func,
                "Gibberish Score": aggregation_func,
                "visited_states": lambda x: set().union(*x),
                "turn": "max",
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


def load_trajs_from_path(traj_path: Path, final_scores: bool = False):
    turns_df = load_turns_df_from_traj_path_custom(traj_path)
    traj_df = group_turns_df_to_traj_df(turns_df, final=final_scores)

    # This merge includes the traj-level reward/influence info to the turns_df entries for wandb logging.
    turns_df = turns_df.merge(traj_df, on=["env_name", "initial_state_id", "trajectory_id"])
    return turns_df, traj_df


def generate_and_preprocess_trajectories(
    run_name: str, final_scores: bool = True, top_n: Optional[int] = None
) -> Generator[pd.DataFrame, None, None]:
    data_path = PROJECT_DATA / "trajectories" / run_name
    iterations = sorted(int(d.name) for d in data_path.iterdir() if d.is_dir() and d.name.isdigit())
    for iteration in iterations:
        iteration_path = data_path / str(iteration)
        _, traj_df = load_trajs_from_path(iteration_path, final_scores)
        if top_n is not None:
            traj_df = get_selected_traj_df(traj_df, num_chosen_trajs=top_n, func=pd.DataFrame.nlargest)

        # Rescale the trajectory reward to the interval [0, 1]
        traj_df["traj_rew"] = traj_df["traj_rew"] / 5

        yield traj_df


def compute_average_trajectory_from_data(
    run_name: str, final_scores: bool = True, top_n: Optional[int] = None, **kwargs
) -> list[tuple[float, float, float, float]]:
    # Compute a bunch of metrics for the run
    trajectory = []

    for traj_df in generate_and_preprocess_trajectories(run_name, final_scores, top_n):

        reward_mu, reward_se = mean_and_stderr(traj_df["traj_rew"])  # type: ignore
        truth_mu, truth_se = mean_and_stderr(traj_df["Truth Score"])
        lie_mu, lie_se = mean_and_stderr(traj_df["Lie Score"])
        gibberish_mu, gibberish_se = mean_and_stderr(traj_df["Gibberish Score"])

        trajectory.append((truth_mu, lie_mu, gibberish_mu, reward_mu))

    # Print the trajectory
    trajectory_rounded = [tuple(round(x, 6) for x in traj) for traj in trajectory]
    trajectory_df = pd.DataFrame(trajectory_rounded, columns=["Truth", "Lie", "Gibberish", "Reward"])
    trajectory_df.index.name = "Iteration"
    print(trajectory_df)

    return trajectory


def load_trajectories_from_data(
    run_name: str, final_scores: bool = True, top_n: Optional[int] = None, **kwargs
) -> list[pd.DataFrame]:
    return list(generate_and_preprocess_trajectories(run_name, final_scores, top_n))


def main(run_name: str, top_n: Optional[int] = None, mode: str = "trajectory", **kwargs):
    supported_modes = {
        "trajectory": (compute_average_trajectory_from_data, plot_trajectory),
        "kde": (load_trajectories_from_data, plot_kernel_density_estimation),
    }

    assert mode in supported_modes, f"Mode {mode} not supported. Choose from {supported_modes.keys()}"

    prepare_data, plot_data = supported_modes[mode]

    # Compute the trajectory from the experiment data
    data = prepare_data(run_name=run_name, top_n=top_n, **kwargs)

    # Plot the trajectory
    plot_data(data)


if __name__ == "__main__":
    # run_name = "kto-lying_doctor-09-13_18-54"
    # run_name = "kto-lying_doctor-09-13_18-54_copy"  # Very cool result!
    # run_name = "kto-lying_doctor-09-13_22-29"
    # run_name = "kto-lying_doctor-09-13_22-49"
    # run_name = "kto-lying_doctor_llama3.1-09-16_00-39"
    # run_name = "kto-lying_doctor_llama3.1-09-16_00-39 copy"
    # run_name = "kto-lying_doctor_llama3.1_round3-09-17_22-29"
    # run_name = "kto-lying_doctor_llama3.1_round4-09-18_03-49"
    # run_name = "kto-lying_doctor_llama3.1_round4-09-18_04-25"
    # run_name = "kto-lying_doctor_llama3.1_round4-09-18_16-07"
    # run_name = "kto-lying_doctor_llama3.1_round5-09-18_22-21-12"
    run_name = "kto-lying_doctor_llama3.1_round5-09-18_22-23-01"

    mode = "kde"
    main(run_name, top_n=None, mode=mode)
