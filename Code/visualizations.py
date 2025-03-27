import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

sns.set_theme(font_scale=200)


# Function to calculate 95% confidence interval
def confidence_interval(data):
    n = len(data)
    mean = np.mean(data)
    std_err = np.std(data, ddof=1) / np.sqrt(n)
    margin_of_error = 1.96 * std_err
    return mean, mean - margin_of_error, mean + margin_of_error

##########################################################################################################################

def visualize_width(csv_file, base_dataset, filter_type="all", data_type="testing"):
    """
    Visualize the average linear separability score and 95% confidence interval for each layer of the network
    based on different base widths.

    Parameters:
    csv_file (str): Path to the CSV file containing the experiment results.
    base_dataset (str): Dataset name.
    filter_type (str): One of ["frozen", "unfrozen", "all"] to filter the data.
    data_type (str): One of ["training", "testing"] to determine which scores to visualize.
    """
    if data_type not in ["training", "testing"]:
        raise ValueError("data_type must be one of ['training', 'testing']")

    df = pd.read_csv(csv_file)

    if filter_type == "frozen":
        df = df[df["frozen"] == True]
    elif filter_type == "unfrozen":
        df = df[df["frozen"] == False]
    elif filter_type != "all":
        raise ValueError("filter_type must be one of ['frozen', 'unfrozen', 'all']")

    score_column_prefix = "layer_{}_train_separability" if data_type == "training" else "layer_{}_test_separability"
    accuracy_column = "train_accuracy" if data_type == "training" else "val_accuracy"

    grouped = df.groupby(['frozen', 'layer_base_width'])
    plot_data = []

    for (frozen, base_width), group in grouped:
        for i in range(6):  # Assuming layers 0 to 5 exist
            layer_column = score_column_prefix.format(i)
            if layer_column in group.columns:
                mean, lower, upper = confidence_interval(group[layer_column])
                plot_data.append({
                    "frozen": frozen,
                    "layer_base_width": base_width,
                    "layer": i,
                    "mean_separability": mean,
                    "lower_ci": lower,
                    "upper_ci": upper
                })

    plot_df = pd.DataFrame(plot_data)

    sns.set(style="whitegrid")
    plt.figure(figsize=(9, 6))
    

    colors = sns.color_palette("plasma", n_colors=plot_df['layer_base_width'].nunique())
    color_map = {width: colors[i] for i, width in enumerate(sorted(plot_df['layer_base_width'].unique()))}

    for (frozen, base_width), group in plot_df.groupby(['frozen', 'layer_base_width']):
        color = color_map[base_width]
        linestyle = 'solid' if frozen else 'dashed'
        plt.plot(group["layer"], group["mean_separability"],
                 color=color, linestyle=linestyle, alpha=1)
        plt.fill_between(group["layer"], group["lower_ci"], group["upper_ci"], color=color, alpha=0.1)

    for layer in range(1, 6, 3):
        plt.axvline(x=layer, color='black', linestyle='dotted', linewidth=1.5)

    plt.xticks(np.arange(0, 6, 1))
    plt.grid(True, which='both', linestyle='dotted', linewidth=0.5)

    plt.xlabel("Layer Index")
    plt.ylabel("Average Linear Separability Score")
    plt.title(f"Average Linear Separability Score ({base_dataset.upper()}, {data_type.capitalize()}) with 95% CI by Network Base Width")

    # Create separate legends
    frozen_legend = [plt.Line2D([0], [0], color='black', linestyle='solid', label="Frozen"),
                     plt.Line2D([0], [0], color='black', linestyle='dashed', label="Fully Trainable")]

    width_legend = [plt.Line2D([0], [0], color=color_map[width], linestyle='solid', label=f"Width={width}")
                    for width in sorted(plot_df['layer_base_width'].unique())]

    # Combine legends side-by-side inside the plot
    legend1 = plt.legend(handles=frozen_legend, loc='lower right', frameon=True, bbox_to_anchor=(0.75, 0))
    plt.gca().add_artist(legend1)
    legend2 = plt.legend(handles=width_legend, loc='lower right', frameon=True, bbox_to_anchor=(1, 0))

    plt.tight_layout()
    filename = f"Results/Images/{base_dataset}_width_{data_type}_{filter_type}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

    # Conduct t-test for each width
    widths = df['layer_base_width'].unique()
    for width in widths:
        frozen_accuracy = df[(df["frozen"] == True) & (df["layer_base_width"] == width)][accuracy_column]
        unfrozen_accuracy = df[(df["frozen"] == False) & (df["layer_base_width"] == width)][accuracy_column]

        if not frozen_accuracy.empty and not unfrozen_accuracy.empty:
            mean_frozen_accuracy = np.mean(frozen_accuracy)
            mean_unfrozen_accuracy = np.mean(unfrozen_accuracy)

            t_stat, p_value = ttest_ind(frozen_accuracy, unfrozen_accuracy, equal_var=False)

            print(f"Width: {width}")
            print(f"  Mean Accuracy (Frozen): {mean_frozen_accuracy}")
            print(f"  Mean Accuracy (Unfrozen): {mean_unfrozen_accuracy}")
            print(f"  P-value for t-test: {p_value}\n")

##########################################################################################################################

def visualize_depth(csv_file, base_dataset, filter_type="all", max_reservoir_layers=None, data_type="testing"):
    """
    Visualize the average linear separability score and 95% confidence interval for each layer of the network,
    with an option to filter networks with fewer than a specified number of reservoir layers.

    Parameters:
    csv_file (str): Path to the CSV file containing the experiment results.
    base_dataset (str): Dataset name.
    filter_type (str): One of ["frozen", "unfrozen", "all"] to filter the data.
    max_reservoir_layers (int, optional): Maximum number of reservoir layers to include.
    data_type (str): One of ["training", "testing"] to specify which data to analyze.
    """
    if data_type not in ["training", "testing"]:
        raise ValueError("data_type must be one of ['training', 'testing']")

    df = pd.read_csv(csv_file)

    if filter_type == "frozen":
        df = df[df["frozen"] == True]
    elif filter_type == "unfrozen":
        df = df[df["frozen"] == False]
    elif filter_type != "all":
        raise ValueError("filter_type must be one of ['frozen', 'unfrozen', 'all']")

    if max_reservoir_layers is not None:
        df = df[df["num_reservoir_layers"] <= max_reservoir_layers]

    score_column_prefix = "layer_{}_train_separability" if data_type == "training" else "layer_{}_test_separability"
    accuracy_column = "train_accuracy" if data_type == "training" else "val_accuracy"

    grouped = df.groupby(['frozen', 'num_reservoir_layers'])
    plot_data = []

    for (frozen, num_layers), group in grouped:
        for i in range(3 * num_layers):
            layer_column = score_column_prefix.format(i)
            if layer_column in group.columns:
                mean, lower, upper = confidence_interval(group[layer_column])
                plot_data.append({
                    "frozen": frozen,
                    "num_reservoir_layers": num_layers,
                    "layer": i,
                    "mean_separability": mean,
                    "lower_ci": lower,
                    "upper_ci": upper
                })

    plot_df = pd.DataFrame(plot_data)

    sns.set(style="whitegrid")
    plt.figure(figsize=(9, 6))
    

    colors = sns.color_palette("plasma", n_colors=plot_df['num_reservoir_layers'].nunique())
    color_map = {depth: colors[i] for i, depth in enumerate(sorted(plot_df['num_reservoir_layers'].unique()))}

    for (frozen, num_layers), group in plot_df.groupby(['frozen', 'num_reservoir_layers']):
        color = color_map[num_layers]
        linestyle = 'solid' if frozen else 'dashed'
        plt.plot(group["layer"], group["mean_separability"],
                 color=color, linestyle=linestyle, alpha=1)
        plt.fill_between(group["layer"], group["lower_ci"], group["upper_ci"], color=color, alpha=0.1)

    for layer in range(1, 3 * max(plot_df['num_reservoir_layers']), 3):
        plt.axvline(x=layer, color='black', linestyle='dotted', linewidth=1.5)

    max_layers = plot_df['num_reservoir_layers'].max()
    plt.xlim(0, 3 * max_layers)
    plt.xticks(np.arange(0, 3 * max_layers + 1, 3))
    plt.grid(True, which='both', linestyle='dotted', linewidth=0.5)

    plt.xlabel("Layer Index")
    plt.ylabel("Average Linear Separability Score")
    plt.title(f"Average Linear Separability Score ({base_dataset.upper()}, {data_type.capitalize()}, Skip Connections) with 95% CI by Network Depth")

    # Create separate legends
    frozen_legend = [plt.Line2D([0], [0], color='black', linestyle='solid', label="Frozen"),
                     plt.Line2D([0], [0], color='black', linestyle='dashed', label="Fully Trainable")]

    depth_legend = [plt.Line2D([0], [0], color=color_map[depth], linestyle='solid', label=f"Blocks={depth}")
                    for depth in sorted(plot_df['num_reservoir_layers'].unique())]

    # Combine legends side-by-side inside the plot
    legend1 = plt.legend(handles=frozen_legend, loc='lower right', frameon=True, bbox_to_anchor=(0.75, 0))
    plt.gca().add_artist(legend1)
    legend2 = plt.legend(handles=depth_legend, loc='lower right', frameon=True, bbox_to_anchor=(1, 0))

    plt.tight_layout()
    filename = f"Results/Images/{base_dataset}_depth_{data_type}_{filter_type}_skip.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

    depths = df['num_reservoir_layers'].unique()
    for depth in depths:
        frozen_accuracy = df[(df["frozen"] == True) & (df["num_reservoir_layers"] == depth)][accuracy_column]
        unfrozen_accuracy = df[(df["frozen"] == False) & (df["num_reservoir_layers"] == depth)][accuracy_column]

        if not frozen_accuracy.empty and not unfrozen_accuracy.empty:
            mean_frozen_accuracy = np.mean(frozen_accuracy)
            mean_unfrozen_accuracy = np.mean(unfrozen_accuracy)

            t_stat, p_value = ttest_ind(frozen_accuracy, unfrozen_accuracy, equal_var=False)

            print(f"Reservoir Blocks: {depth}")
            print(f"  Mean Accuracy (Frozen): {mean_frozen_accuracy}")
            print(f"  Mean Accuracy (Unfrozen): {mean_unfrozen_accuracy}")
            print(f"  P-value for t-test: {p_value}\n")




##########################################################################################################################

def visualize_scaling(csv_file, base_dataset, filter_type="all", data_type="testing"):
    """
    Visualize the effect of reservoir layer scaling factor on average linear separability score.

    Parameters:
    csv_file (str): Path to the CSV file containing the experiment results.
    base_dataset (str): Dataset name.
    filter_type (str): One of ["frozen", "unfrozen", "all"] to filter the data.
    data_type (str): One of ["training", "testing"] to specify which data to analyze.
    """
    if data_type not in ["training", "testing"]:
        raise ValueError("data_type must be one of ['training', 'testing']")

    df = pd.read_csv(csv_file)

    if filter_type == "frozen":
        df = df[df["frozen"] == True]
    elif filter_type == "unfrozen":
        df = df[df["frozen"] == False]
    elif filter_type != "all":
        raise ValueError("filter_type must be one of ['frozen', 'unfrozen', 'all']")

    score_column_prefix = "layer_{}_train_separability" if data_type == "training" else "layer_{}_test_separability"
    accuracy_column = "train_accuracy" if data_type == "training" else "val_accuracy"

    grouped = df.groupby(['reservoir_layer_scaling_factor', 'frozen'])
    plot_data = []

    for (scaling_factor, frozen), group in grouped:
        for i in range(6):  # Assuming 6 total layers
            layer_column = score_column_prefix.format(i)
            if layer_column in group.columns:
                mean, lower, upper = confidence_interval(group[layer_column])
                plot_data.append({
                    "scaling_factor": scaling_factor,
                    "frozen": frozen,
                    "layer": i,
                    "mean_separability": mean,
                    "lower_ci": lower,
                    "upper_ci": upper
                })

    plot_df = pd.DataFrame(plot_data)

    sns.set(style="whitegrid")
    plt.figure(figsize=(9, 6))
    

    colors = sns.color_palette("plasma", n_colors=plot_df['scaling_factor'].nunique())
    color_map = {factor: colors[i] for i, factor in enumerate(sorted(plot_df['scaling_factor'].unique()))}

    for (scaling_factor, frozen), group in plot_df.groupby(['scaling_factor', 'frozen']):
        color = color_map[scaling_factor]
        linestyle = 'solid' if frozen else 'dashed'
        plt.plot(group["layer"], group["mean_separability"],
                 color=color, linestyle=linestyle, alpha=1)
        plt.fill_between(group["layer"], group["lower_ci"], group["upper_ci"], color=color, alpha=0.1)

    for layer in range(1, 6, 3):
        plt.axvline(x=layer, color='black', linestyle='dotted', linewidth=1.5)

    plt.xlim(0, 5)
    plt.xticks(np.arange(0, 6, 1))
    plt.grid(True, which='both', linestyle='dotted', linewidth=0.5)

    plt.xlabel("Layer Index")
    plt.ylabel("Average Linear Separability Score")
    plt.title(f"Average Linear Separability Score ({base_dataset.upper()}, {data_type.capitalize()}) with 95% CI by Reservoir Scaling Factor")

    # Create separate legends
    frozen_legend = [plt.Line2D([0], [0], color='black', linestyle='solid', label="Frozen"),
                     plt.Line2D([0], [0], color='black', linestyle='dashed', label="Fully Trainable")]

    scaling_legend = [plt.Line2D([0], [0], color=color_map[factor], linestyle='solid', label=f"Scaling={factor}")
                      for factor in sorted(plot_df['scaling_factor'].unique())]

    # Combine legends side-by-side inside the plot
    legend1 = plt.legend(handles=frozen_legend, loc='lower right', frameon=True, bbox_to_anchor=(0.75, 0))
    plt.gca().add_artist(legend1)
    legend2 = plt.legend(handles=scaling_legend, loc='lower right', frameon=True, bbox_to_anchor=(1, 0))

    plt.tight_layout()
    filename = f"Results/Images/{base_dataset}_scaling_{data_type}_{filter_type}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

    scalings = df['reservoir_layer_scaling_factor'].unique()
    for scaling in scalings:
        frozen_accuracy = df[(df["frozen"] == True) & (df["reservoir_layer_scaling_factor"] == scaling)][accuracy_column]
        unfrozen_accuracy = df[(df["frozen"] == False) & (df["reservoir_layer_scaling_factor"] == scaling)][accuracy_column]

        if not frozen_accuracy.empty and not unfrozen_accuracy.empty:
            mean_frozen_accuracy = np.mean(frozen_accuracy)
            mean_unfrozen_accuracy = np.mean(unfrozen_accuracy)

            t_stat, p_value = ttest_ind(frozen_accuracy, unfrozen_accuracy, equal_var=False)

            print(f"Scaling Factor: {scaling}")
            print(f"  Mean Accuracy (Frozen): {mean_frozen_accuracy}")
            print(f"  Mean Accuracy (Unfrozen): {mean_unfrozen_accuracy}")
            print(f"  P-value for t-test: {p_value}\n")



##########################################################################################################################

def visualize_position(csv_file, base_dataset, filter_type="all", data_type="testing"):
    """
    Visualize the average linear separability score and 95% confidence interval for each layer position.

    Parameters:
    csv_file (str): Path to the CSV file containing the experiment results.
    base_dataset (str): Dataset name.
    filter_type (str): One of ["frozen", "unfrozen", "all"] to filter the data.
    data_type (str): One of ["training", "testing"] to specify which data to analyze.
    """
    if data_type not in ["training", "testing"]:
        raise ValueError("data_type must be one of ['training', 'testing']")

    df = pd.read_csv(csv_file)

    if filter_type == "frozen":
        df = df[df["frozen"] == True]
    elif filter_type == "unfrozen":
        df = df[df["frozen"] == False]
    elif filter_type != "all":
        raise ValueError("filter_type must be one of ['frozen', 'unfrozen', 'all']")

    score_column_prefix = "layer_{}_train_separability" if data_type == "training" else "layer_{}_test_separability"
    accuracy_column = "train_accuracy" if data_type == "training" else "val_accuracy"

    grouped = df.groupby(['frozen', 'position'])
    plot_data = []

    for (frozen, position), group in grouped:
        for i in range(6):  # Assuming 6 total layers
            layer_column = score_column_prefix.format(i)
            if layer_column in group.columns:
                mean, lower, upper = confidence_interval(group[layer_column])
                plot_data.append({
                    "frozen": frozen,
                    "position": position,
                    "layer": i,
                    "mean_separability": mean,
                    "lower_ci": lower,
                    "upper_ci": upper
                })

    plot_df = pd.DataFrame(plot_data)

    sns.set(style="whitegrid")
    plt.figure(figsize=(9, 6))
    

    colors = sns.color_palette("plasma", n_colors=plot_df['position'].nunique())
    color_map = {position: colors[i] for i, position in enumerate(sorted(plot_df['position'].unique()))}

    for (frozen, position), group in plot_df.groupby(['frozen', 'position']):
        color = color_map[position]
        linestyle = 'solid' if frozen else 'dashed'
        plt.plot(group["layer"], group["mean_separability"],
                 color=color, linestyle=linestyle, alpha=1)
        plt.fill_between(group["layer"], group["lower_ci"], group["upper_ci"], color=color, alpha=0.1)

    for layer in range(1, 6, 3):
        plt.axvline(x=layer, color='black', linestyle='dotted', linewidth=1.5)

    plt.xlim(0, 5)
    plt.xticks(np.arange(0, 6, 1))
    plt.grid(True, which='both', linestyle='dotted', linewidth=0.5)

    plt.xlabel("Layer Index")
    plt.ylabel("Average Linear Separability Score")
    plt.title(f"Average Linear Separability Score ({base_dataset.upper()}, {data_type.capitalize()}) with 95% CI by Reservoir Layer Position")

    # Create separate legends
    frozen_legend = [plt.Line2D([0], [0], color='black', linestyle='solid', label="Frozen"),
                     plt.Line2D([0], [0], color='black', linestyle='dashed', label="Fully Trainable")]

    position_legend = [plt.Line2D([0], [0], color=color_map[position], linestyle='solid', label=f"Position={position}")
                       for position in sorted(plot_df['position'].unique())]

    # Combine legends side-by-side inside the plot
    legend1 = plt.legend(handles=frozen_legend, loc='lower right', frameon=True, bbox_to_anchor=(0.75, 0))
    plt.gca().add_artist(legend1)
    legend2 = plt.legend(handles=position_legend, loc='lower right', frameon=True, bbox_to_anchor=(1, 0))

    plt.tight_layout()
    filename = f"Results/Images/{base_dataset}_position_{data_type}_{filter_type}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

    positions = df['position'].unique()
    for position in positions:
        frozen_accuracy = df[(df["frozen"] == True) & (df["position"] == position)][accuracy_column]
        unfrozen_accuracy = df[(df["frozen"] == False) & (df["position"] == position)][accuracy_column]

        if not frozen_accuracy.empty and not unfrozen_accuracy.empty:
            mean_frozen_accuracy = np.mean(frozen_accuracy)
            mean_unfrozen_accuracy = np.mean(unfrozen_accuracy)

            t_stat, p_value = ttest_ind(frozen_accuracy, unfrozen_accuracy, equal_var=False)

            print(f"Reservoir Position: {position}")
            print(f"  Mean Accuracy (Frozen): {mean_frozen_accuracy}")
            print(f"  Mean Accuracy (Unfrozen): {mean_unfrozen_accuracy}")
            print(f"  P-value for t-test: {p_value}\n")

##########################################################################################################################

def visualize_regularization(csv_file, base_dataset, filter_type="all", data_type="testing"):
    """
    Visualize the impact of regularization on the average linear separability score with 95% confidence interval.

    Parameters:
    csv_file (str): Path to the CSV file containing the experiment results.
    base_dataset (str): Dataset name.
    filter_type (str): One of ["frozen", "unfrozen", "all"] to filter the data.
    data_type (str): One of ["training", "testing"] to specify which data to analyze.
    """
    if data_type not in ["training", "testing"]:
        raise ValueError("data_type must be one of ['training', 'testing']")

    df = pd.read_csv(csv_file)

    if filter_type == "frozen":
        df = df[df["frozen"] == True]
    elif filter_type == "unfrozen":
        df = df[df["frozen"] == False]
    elif filter_type != "all":
        raise ValueError("filter_type must be one of ['frozen', 'unfrozen', 'all']")

    score_column_prefix = "layer_{}_train_separability" if data_type == "training" else "layer_{}_test_separability"
    accuracy_column = "train_accuracy" if data_type == "training" else "val_accuracy"

    # Parse the regularization column
    df['regularization_type'] = df['regularization'].apply(lambda x: x.split('_')[0])
    df['regularization_amount'] = df['regularization'].apply(lambda x: float(x.split('_')[1]) if len(x.split('_')) > 1 else 0)

    grouped = df.groupby(['frozen', 'regularization_type', 'regularization_amount'])
    plot_data = []

    for (frozen, reg_type, reg_amount), group in grouped:
        for i in range(6):  # Assuming 6 total layers
            layer_column = score_column_prefix.format(i)
            if layer_column in group.columns:
                mean, lower, upper = confidence_interval(group[layer_column])
                plot_data.append({
                    "frozen": frozen,
                    "regularization_type": reg_type,
                    "regularization_amount": reg_amount,
                    "layer": i,
                    "mean_separability": mean,
                    "lower_ci": lower,
                    "upper_ci": upper
                })

    plot_df = pd.DataFrame(plot_data)

    sns.set(style="whitegrid")
    plt.figure(figsize=(9, 6))
    plt.rcParams.update({'font.size': 14})  # Adjust the size as needed

    colors = sns.color_palette("plasma", n_colors=plot_df['regularization_type'].nunique())
    color_map = {reg_type: colors[i] for i, reg_type in enumerate(sorted(plot_df['regularization_type'].unique()))}

    for (frozen, reg_type, reg_amount), group in plot_df.groupby(['frozen', 'regularization_type', 'regularization_amount']):
        color = color_map[reg_type]
        linestyle = 'solid' if frozen else 'dashed'
        label = f"{reg_type.capitalize()} ({reg_amount})"
        plt.plot(group["layer"], group["mean_separability"],
                 label=label,
                 color=color, linestyle=linestyle, alpha=1)
        plt.fill_between(group["layer"], group["lower_ci"], group["upper_ci"], color=color, alpha=0.1)

    for layer in range(1, 6, 3):
        plt.axvline(x=layer, color='black', linestyle='dotted', linewidth=1.5)

    plt.xlim(0, 5)
    plt.xticks(np.arange(0, 6, 1))
    plt.grid(True, which='both', linestyle='dotted', linewidth=0.5)

    plt.xlabel("Layer Index")
    plt.ylabel("Average Linear Separability Score")
    plt.title(f"Average Linear Separability Score ({base_dataset.upper()}, {data_type.capitalize()}) with 95% CI by Regularization Strategy")

    # Create separate legends
    frozen_legend = [plt.Line2D([0], [0], color='black', linestyle='solid', label="Frozen"),
                     plt.Line2D([0], [0], color='black', linestyle='dashed', label="Fully Trainable")]

    # Sort the DataFrame by regularization type and amount
    sorted_reg_df = plot_df[['regularization_type', 'regularization_amount']].drop_duplicates().sort_values(by=['regularization_type', 'regularization_amount'])
    regularization_legend = [plt.Line2D([0], [0], color=color_map[row['regularization_type']], linestyle='solid',
                                         label=f"{row['regularization_type'].capitalize()} ({row['regularization_amount']})")
                             for _, row in sorted_reg_df.iterrows()]

    # Combine legends side-by-side inside the plot
    legend1 = plt.legend(handles=frozen_legend, loc='lower left', frameon=True, bbox_to_anchor=(0.25, 0))
    plt.gca().add_artist(legend1)
    legend2 = plt.legend(handles=regularization_legend, loc='lower left', frameon=True, bbox_to_anchor=(0, 0))

    plt.tight_layout()
    filename = f"Results/Images/{base_dataset}_regularization_{data_type}_{filter_type}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

    # Conduct t-test for each regularization type and amount
    regularizations = df[['regularization_type', 'regularization_amount']].drop_duplicates()
    for _, row in regularizations.iterrows():
        reg_type = row['regularization_type']
        reg_amount = row['regularization_amount']
        frozen_accuracy = df[(df["frozen"] == True) &
                             (df["regularization_type"] == reg_type) &
                             (df["regularization_amount"] == reg_amount)][accuracy_column]
        unfrozen_accuracy = df[(df["frozen"] == False) &
                               (df["regularization_type"] == reg_type) &
                               (df["regularization_amount"] == reg_amount)][accuracy_column]

        if not frozen_accuracy.empty and not unfrozen_accuracy.empty:
            mean_frozen_accuracy = np.mean(frozen_accuracy)
            mean_unfrozen_accuracy = np.mean(unfrozen_accuracy)

            t_stat, p_value = ttest_ind(frozen_accuracy, unfrozen_accuracy, equal_var=False)

            print(f"Regularization: {reg_type} (Amount: {reg_amount})")
            print(f"  Mean Accuracy (Frozen): {mean_frozen_accuracy}")
            print(f"  Mean Accuracy (Unfrozen): {mean_unfrozen_accuracy}")
            print(f"  P-value for t-test: {p_value}\n")

##########################################################################################################################

def visualize_linear_separability(csv_file, base_dataset, filter_type="all"):
    """
    Visualize the linear separability of hidden state features with 95% confidence interval.
    """
    df = pd.read_csv(csv_file)
    if filter_type == "frozen":
        df = df[df["frozen"] == True]
    elif filter_type == "unfrozen":
        df = df[df["frozen"] == False]
    elif filter_type != "all":
        raise ValueError("filter_type must be one of ['frozen', 'unfrozen', 'all']")

    score_columns = {
        "Training Features (Train Classifier)": "layer_{}_training_separability_train",
        "Testing Features (Train Classifier)": "layer_{}_testing_separability_train",
        "Testing Features (Test Classifier)": "layer_{}_testing_separability_test",
        "Training Features (Test Classifier)": "layer_{}_training_separability_test"
    }

    plot_data = []
    for i in range(6):  # Assuming 6 total layers
        for label, score_column_template in score_columns.items():
            layer_column = score_column_template.format(i)
            if layer_column in df.columns:
                grouped = df.groupby('frozen')
                for frozen, group in grouped:
                    mean, lower, upper = confidence_interval(group[layer_column])
                    plot_data.append({
                        "frozen": bool(frozen),  # Convert tuple to boolean
                        "layer": i,
                        "mean_separability": mean,
                        "lower_ci": lower,
                        "upper_ci": upper,
                        "score_type": label
                    })

    plot_df = pd.DataFrame(plot_data)
    sns.set(style="whitegrid")
    plt.figure(figsize=(9, 6))
    
    colors = sns.color_palette("plasma", n_colors=plot_df['score_type'].nunique())
    color_map = {score_type: colors[i] for i, score_type in enumerate(sorted(plot_df['score_type'].unique()))}

    for (frozen, score_type), group in plot_df.groupby(['frozen', 'score_type']):
        color = color_map[score_type]
        linestyle = 'solid' if frozen else 'dashed'
        plt.plot(group["layer"], group["mean_separability"],
                 color=color, linestyle=linestyle, alpha=1)
        plt.fill_between(group["layer"], group["lower_ci"], group["upper_ci"], color=color, alpha=0.1)

    for layer in range(1, 6, 3):
        plt.axvline(x=layer, color='black', linestyle='dotted', linewidth=1.5)

    plt.xlim(0, 5)
    plt.xticks(np.arange(0, 6, 1))
    plt.grid(True, which='both', linestyle='dotted', linewidth=0.5)

    plt.xlabel("Layer Index")
    plt.ylabel("Average Linear Separability Score")
    plt.title(f"Average Linear Separability Score ({base_dataset.upper()}) with 95% CI")

    # Create separate legends
    frozen_legend = [plt.Line2D([0], [0], color='black', linestyle='solid', label="Frozen"),
                     plt.Line2D([0], [0], color='black', linestyle='dashed', label="Fully Trainable")]

    score_type_legend = [plt.Line2D([0], [0], color=color_map[score_type], linestyle='solid', label=f"{score_type}")
                         for score_type in sorted(plot_df['score_type'].unique())]

    # Combine legends side-by-side inside the plot
    legend1 = plt.legend(handles=frozen_legend, loc='lower right', frameon=True, bbox_to_anchor=(0.6, 0))
    plt.gca().add_artist(legend1)
    legend2 = plt.legend(handles=score_type_legend, loc='lower right', frameon=True, bbox_to_anchor=(1, 0))

    plt.tight_layout()
    filename = f"Results/Images/{base_dataset}_linear_separability_{filter_type}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

##########################################################################################################################

def visualize_training(csv_file, base_dataset, filter_type="all", data_type="testing", num_epochs_to_plot=1):
    """
    Visualize the average linear separability score and 95% confidence interval for each layer of the network
    based on the first specified number of epochs.

    Parameters:
    csv_file (str): Path to the CSV file containing the experiment results.
    base_dataset (str): Dataset name.
    filter_type (str): One of ["frozen", "unfrozen", "all"] to filter the data.
    data_type (str): One of ["training", "testing"] to specify which data to analyze.
    num_epochs_to_plot (int): Number of the first epochs to plot.
    """
    if data_type not in ["training", "testing"]:
        raise ValueError("data_type must be one of ['training', 'testing']")

    df = pd.read_csv(csv_file)

    if filter_type == "frozen":
        df = df[df["frozen"] == True]
    elif filter_type == "unfrozen":
        df = df[df["frozen"] == False]
    elif filter_type != "all":
        raise ValueError("filter_type must be one of ['frozen', 'unfrozen', 'all']")

    score_column_prefix = "layer_{}_train_separability" if data_type == "training" else "layer_{}_test_separability"
    accuracy_column = "train_accuracy" if data_type == "training" else "val_accuracy"

    grouped = df.groupby(['frozen', 'epoch'])
    plot_data = []

    for (frozen, epoch), group in grouped:
        if epoch < num_epochs_to_plot:  # Plot only the first num_epochs_to_plot epochs
            for i in range(6):
                layer_column = score_column_prefix.format(i)
                if layer_column in group.columns:
                    mean, lower, upper = confidence_interval(group[layer_column])
                    plot_data.append({
                        "frozen": frozen,
                        "epoch": epoch,
                        "layer": i,
                        "mean_separability": mean,
                        "lower_ci": lower,
                        "upper_ci": upper
                    })

    plot_df = pd.DataFrame(plot_data)

    sns.set(style="whitegrid")
    plt.figure(figsize=(9, 6))
    

    colors = sns.color_palette("plasma", n_colors=plot_df['epoch'].nunique())
    color_map = {epoch: colors[i] for i, epoch in enumerate(sorted(plot_df['epoch'].unique()))}

    for (frozen, epoch), group in plot_df.groupby(['frozen', 'epoch']):
        color = color_map[epoch]
        linestyle = 'solid' if frozen else 'dashed'
        plt.plot(group["layer"], group["mean_separability"],
                 color=color, linestyle=linestyle, alpha=1)
        plt.fill_between(group["layer"], group["lower_ci"], group["upper_ci"], color=color, alpha=0.1)

    for layer in range(1, 6, 3):
        plt.axvline(x=layer, color='black', linestyle='dotted', linewidth=1.5)

    plt.xticks(np.arange(0, 6, 1))
    plt.grid(True, which='both', linestyle='dotted', linewidth=0.5)

    plt.xlabel("Layer Index")
    plt.ylabel("Average Linear Separability Score")
    plt.title(f"Average Linear Separability Score ({base_dataset.upper()}, {data_type.capitalize()}) with 95% CI by Epoch")

    # Create separate legends
    frozen_legend = [plt.Line2D([0], [0], color='black', linestyle='solid', label="Frozen"),
                     plt.Line2D([0], [0], color='black', linestyle='dashed', label="Fully Trainable")]

    epoch_legend = [plt.Line2D([0], [0], color=color_map[epoch], linestyle='solid', label=f"Epoch={epoch}")
                    for epoch in sorted(plot_df['epoch'].unique())]

    # Combine legends side-by-side inside the plot
    legend1 = plt.legend(handles=frozen_legend, loc='lower left', frameon=True, bbox_to_anchor=(0.25, 0))
    plt.gca().add_artist(legend1)
    legend2 = plt.legend(handles=epoch_legend, loc='lower left', frameon=True, bbox_to_anchor=(0, 0))

    plt.tight_layout()
    filename = f"Results/Images/{base_dataset}_training_{data_type}_{filter_type}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

    # Conduct t-test for each epoch
    epochs = df['epoch'].unique()
    for epoch in epochs:
        frozen_accuracy = df[(df["frozen"] == True) & (df["epoch"] == epoch)][accuracy_column]
        unfrozen_accuracy = df[(df["frozen"] == False) & (df["epoch"] == epoch)][accuracy_column]

        if not frozen_accuracy.empty and not unfrozen_accuracy.empty:
            mean_frozen_accuracy = np.mean(frozen_accuracy)
            mean_unfrozen_accuracy = np.mean(unfrozen_accuracy)

            t_stat, p_value = ttest_ind(frozen_accuracy, unfrozen_accuracy, equal_var=False)

            print(f"Epoch: {epoch}")
            print(f"  Mean Accuracy (Frozen): {mean_frozen_accuracy}")
            print(f"  Mean Accuracy (Unfrozen): {mean_unfrozen_accuracy}")
            print(f"  P-value for t-test: {p_value}\n")


##########################################################################################################################

def visualize_combined(csv_layer, csv_ratio, base_dataset, layer_index, num_epochs_to_plot=50):
    df_layer = pd.read_csv(csv_layer)
    df_ratio = pd.read_csv(csv_ratio)
    df_ratio = df_ratio[df_ratio["freeze_ratio"].isin([0.1, 0.2, 0.3, 0.4, 0.5])]

    df_layer["freezing_type"] = df_layer["frozen"].map({True: "Layer Frozen", False: "Unfrozen"})
    df_ratio["freezing_type"] = df_ratio["freeze_ratio"].apply(lambda x: f"{int(x * 100)}% Weights Frozen")

    df_combined = pd.concat([df_layer, df_ratio], ignore_index=True)
    grouped = df_combined.groupby(['freezing_type', 'epoch'])

    plot_data = []
    for (freezing_type, epoch), group in grouped:
        if epoch < num_epochs_to_plot:
            layer_column = f"layer_{layer_index}_test_separability"
            if layer_column in group.columns:
                mean, lower, upper = confidence_interval(group[layer_column])
                plot_data.append({
                    "freezing_type": freezing_type,
                    "epoch": epoch,
                    "mean_separability": mean,
                    "lower_ci": lower,
                    "upper_ci": upper
                })

    plot_df = pd.DataFrame(plot_data)

    sns.set(style="whitegrid")
    plt.figure(figsize=(9, 6))
    colors = sns.color_palette("plasma", n_colors=plot_df['freezing_type'].nunique())
    color_map = {freezing_type: colors[i] for i, freezing_type in enumerate(sorted(plot_df['freezing_type'].unique()))}

    for freezing_type, group in plot_df.groupby('freezing_type'):
        color = color_map[freezing_type]
        plt.plot(group["epoch"], group["mean_separability"], color=color, alpha=1, label=freezing_type)
        plt.fill_between(group["epoch"], group["lower_ci"], group["upper_ci"], color=color, alpha=0.1)

    # Set major ticks every 10 epochs
    plt.xticks(np.arange(0, num_epochs_to_plot, 10))

    # Set minor ticks every 5 epochs
    plt.minorticks_on()
    plt.grid(True, which='both', linestyle='dotted', linewidth=0.5)
    plt.grid(True, which='minor', linestyle='dotted', linewidth=0.5, color='lightgrey')

    plt.xlabel("Epoch")
    plt.ylabel("Average Linear Separability Score")
    plt.title(f"Average Linear Separability Score ({base_dataset.upper()}, Layer {layer_index}) with 95% CI by Epoch")

    plt.legend(loc='lower right', frameon=True, bbox_to_anchor=(1, 0))
    plt.tight_layout()
    filename = f"Results/Images/{base_dataset}_layer_{layer_index}_comparison.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

##########################################################################################################################

base_dataset = "cifar_10"

csv_file_path = f"Results/{base_dataset}_width_new.csv"
visualize_width(csv_file_path, base_dataset, filter_type="all", data_type="training")
visualize_width(csv_file_path, base_dataset, filter_type="all", data_type="testing")


csv_file_path = f"Results/{base_dataset}_depth_skip_new.csv"
csv_file_path = f"Results/{base_dataset}_depth_skip_new.csv"
visualize_depth(csv_file_path, base_dataset, filter_type="all", max_reservoir_layers=8, data_type="training")
visualize_depth(csv_file_path, base_dataset, filter_type="all", max_reservoir_layers=8, data_type="testing")


csv_file_path = f"Results/{base_dataset}_scaling_new.csv"
visualize_scaling(csv_file_path, base_dataset, filter_type="all", data_type="training")
visualize_scaling(csv_file_path, base_dataset, filter_type="all", data_type="testing")


csv_file_path = f"Results/{base_dataset}_position_new.csv"
visualize_position(csv_file_path, base_dataset, filter_type="all", data_type="training")
visualize_position(csv_file_path, base_dataset, filter_type="all", data_type="testing")

csv_file_path = f"Results/{base_dataset}_regularization_new.csv"
visualize_regularization(csv_file_path, base_dataset, filter_type="all", data_type="training")
visualize_regularization(csv_file_path, base_dataset, filter_type="all", data_type="testing")

csv_file_path = f"Results/{base_dataset}_frozen_states.csv"
visualize_linear_separability(csv_file_path, base_dataset, filter_type="all")

csv_file_path = f"Results/{base_dataset}_training_separability_new.csv"
visualize_training(csv_file_path, base_dataset, num_epochs_to_plot=5, data_type="training")
visualize_training(csv_file_path, base_dataset, num_epochs_to_plot=5, data_type="testing")


visualize_combined(f"Results/{base_dataset}_training_separability_new.csv", f"Results/{base_dataset}_partial_freezing_separability.csv", base_dataset, layer_index=0, num_epochs_to_plot=11)
visualize_combined(f"Results/{base_dataset}_training_separability_new.csv", f"Results/{base_dataset}_partial_freezing_separability.csv", base_dataset, layer_index=1, num_epochs_to_plot=11)
visualize_combined(f"Results/{base_dataset}_training_separability_new.csv", f"Results/{base_dataset}_partial_freezing_separability.csv", base_dataset, layer_index=2, num_epochs_to_plot=11)
visualize_combined(f"Results/{base_dataset}_training_separability_new.csv", f"Results/{base_dataset}_partial_freezing_separability.csv", base_dataset, layer_index=3, num_epochs_to_plot=11)
visualize_combined(f"Results/{base_dataset}_training_separability_new.csv", f"Results/{base_dataset}_partial_freezing_separability.csv", base_dataset, layer_index=4, num_epochs_to_plot=11)
visualize_combined(f"Results/{base_dataset}_training_separability_new.csv", f"Results/{base_dataset}_partial_freezing_separability.csv", base_dataset, layer_index=5, num_epochs_to_plot=11)


