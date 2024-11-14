import matplotlib
import matplotlib.pyplot as plt
import numpy as np

data = {
    "Reward": {
        "means": [
            1.932436939467126,
            3.1604112587652753,
            3.913814524493611,
            4.354540713569463,
            4.555564373347882,
            4.706121373592044,
            4.703210658233212,
            4.755442223493475,
            4.773327452114608,
            4.786429630546862,
            4.80714831752045,
            4.807978387031864,
            4.807735461713988,
            4.83622627075386,
            4.858675184455034,
        ],
        "stds": [
            0.03394890413248955,
            0.05245601818318965,
            0.03734253441040613,
            0.024758733692813784,
            0.023470696758112867,
            0.021375031991143222,
            0.018680412380023124,
            0.015277798080958999,
            0.025417014739570767,
            0.018107330771522982,
            0.012963880570188624,
            0.01860907829167298,
            0.016619986747854584,
            0.013535369339974799,
            0.015210580374098033,
        ],
    },
    "Omission": {
        "means": [
            1.0,
            1.3375,
            2.575,
            5.8375,
            3.86875,
            4.6,
            3.30625,
            4.20625,
            3.025,
            2.4625,
            2.63125,
            3.19375,
            2.2375,
            1.675,
            1.9,
        ],
        "stds": [
            0.0,
            0.13560008139920746,
            0.271200162798415,
            0.3558680922148033,
            0.3326000434682688,
            0.3496629104486151,
            0.3115942061278624,
            0.3418062540829069,
            0.2980478940871942,
            0.2633073836828105,
            0.27495309419869046,
            0.30644246936138253,
            0.2457960210464444,
            0.18799463059373653,
            0.2141239281438989,
        ],
    },
    "Nudge Not To Book": {
        "means": [
            1.05625,
            3.3625,
            6.68125,
            1.39375,
            1.39375,
            1.05625,
            1.1125,
            1.0,
            1.0,
            1.0,
            1.0,
            1.05625,
            1.05625,
            1.0,
            1.0,
        ],
        "stds": [
            0.05625,
            0.3140433586816882,
            0.3443583810152901,
            0.14598851806104607,
            0.14598851806104607,
            0.05625000000000001,
            0.07929896261426193,
            0.0,
            0.0,
            0.0,
            0.0,
            0.05625,
            0.05625,
            0.0,
            0.0,
        ],
    },
    "Lying Booking Success": {
        "means": [
            1.0,
            1.225,
            1.5625,
            2.40625,
            4.99375,
            5.33125,
            7.24375,
            6.45625,
            7.525,
            7.91875,
            7.91875,
            7.01875,
            8.25625,
            8.875,
            8.875,
        ],
        "stds": [
            0.0,
            0.11143362522232092,
            0.17277050160954877,
            0.25915575241432315,
            0.35460768416633515,
            0.35662219881644946,
            0.3289906035390389,
            0.34872261000172483,
            0.31869797022638074,
            0.30093824391549734,
            0.30093824391549734,
            0.3359338662594172,
            0.28209777886332715,
            0.23604944317375862,
            0.23604944317375862,
        ],
    },
    "Mention Error": {
        "means": [
            9.94375,
            7.075,
            2.06875,
            3.30625,
            2.6875,
            2.0125,
            1.3375,
            1.3375,
            1.45,
            1.61875,
            1.45,
            1.73125,
            1.39375,
            1.45,
            1.225,
        ],
        "stds": [
            0.05625,
            0.3343008756669627,
            0.23089276333479436,
            0.3115942061278624,
            0.2785840630558023,
            0.22553003607016806,
            0.13560008139920746,
            0.13560008139920746,
            0.15555742736220335,
            0.18059820996730894,
            0.15555742736220335,
            0.19500884413906164,
            0.14598851806104604,
            0.15555742736220335,
            0.11143362522232092,
        ],
    },
    "Other": {
        "means": [1.0, 1.0, 1.1125, 1.05625, 1.05625, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.05625, 1.0, 1.0],
        "stds": [
            0.0,
            0.0,
            0.07929896261426193,
            0.05625,
            0.05625000000000001,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.05625,
            0.0,
            0.0,
        ],
    },
}
gradient = np.linspace(0, 1, 15)
colors = matplotlib.colormaps["seismic"](gradient)


def parallel_coordinates_plot(data: dict[str, dict[str, list[float]]]) -> None:
    # Extract means and standard deviations
    means = np.array([v["means"] for v in data.values()])
    stds = np.array([v["stds"] for v in data.values()])

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot each line
    for i in range(means.shape[1]):
        ax.plot(means[:, i], label=f"Iteration {i+1}", color=colors[i])

        # Add confidence intervals
        ax.fill_between(
            range(means.shape[0]), means[:, i] - stds[:, i], means[:, i] + stds[:, i], alpha=0.2, color=colors[i]
        )

    # Customize the plot
    ax.set_xticks(range(len(data)))
    ax.set_xticklabels(data.keys())
    ax.set_xlabel("Variables")
    ax.set_ylabel("Means")
    ax.set_title("Parallel Coordinates Plot with Confidence Intervals")
    ax.legend(loc="upper left")
    ax.grid(True)

    # Adjust layout to prevent legend from being cut off
    plt.tight_layout()
    plt.savefig("targeted_llm_manipulation/stats/parallel_coordinates_plot.png", dpi=300, bbox_inches="tight")
    plt.close()


def radar_chart_plot(data: dict[str, dict[str, list[float]]]):
    # Extract categories and data
    categories = list(data.keys())
    num_categories = len(categories)
    num_variables = len(data[categories[0]]["means"])

    # Set up the angles for the radar chart
    angles = np.linspace(0, 2 * np.pi, num_categories, endpoint=False)

    # Close the plot by appending the first angle to the end
    angles = np.concatenate((angles, [angles[0]]))

    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection="polar"))

    # Plot each variable
    for i in range(num_variables):
        values = [data[cat]["means"][i] for cat in categories]
        values += [values[0]]  # Close the polygon

        ax.plot(angles, values, "o-", linewidth=2, label=f"Variable {i+1}", color=colors[i])

        # Plot confidence intervals
        stds = [data[cat]["stds"][i] for cat in categories]
        stds += [stds[0]]  # Close the polygon
        upper = np.array(values) + np.array(stds)
        lower = np.array(values) - np.array(stds)
        ax.fill_between(angles, lower, upper, alpha=0.1, color=colors[i])

    # Set the labels and title
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_title("Radar Chart")

    # Add legend
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))

    # Adjust the layout and display the plot
    plt.tight_layout()
    plt.savefig("targeted_llm_manipulation/stats/radar_chart_plot.png", dpi=300, bbox_inches="tight")
    plt.close()


def inverse_radar_plot(data: dict[str, dict[str, list[float]]]):
    # Extract categories and data
    categories = list(data.keys())
    num_angles = len(data[categories[0]]["means"])

    # Set up the angles for the radar chart
    angles = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)

    # Close the plot by appending the first angle to the end
    angles = np.concatenate((angles, [angles[0]]))

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection="polar"))

    # Color cycle for different categories
    colors = plt.cm.rainbow(np.linspace(0, 1, len(categories)))

    # Plot each category
    for cat, color in zip(categories, colors):
        values = data[cat]["means"]
        values += [values[0]]  # Close the polygon

        ax.plot(angles, values, "o-", linewidth=2, label=cat, color=color)

        # Plot confidence intervals
        stds = data[cat]["stds"]
        stds += [stds[0]]  # Close the polygon
        upper = np.array(values) + np.array(stds)
        lower = np.array(values) - np.array(stds)
        ax.fill_between(angles, lower, upper, alpha=0.1, color=color)

    # Set the labels and title
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([f"Iteration {i+1}" for i in range(num_angles)])
    ax.set_title("Radar Chart by Category")

    # Add legend
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))

    # Adjust the layout and display the plot
    plt.tight_layout()
    plt.savefig("targeted_llm_manipulation/stats/inverse_radar_chart_plot.png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    # parallel_coordinates_plot(data)
    # radar_chart_plot(data)
    inverse_radar_plot(data)
