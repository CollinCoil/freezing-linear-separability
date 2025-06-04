import os
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from create_network import create_fc_network 
from separability import evaluate_linear_separability, evaluate_linear_separability_mlp
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


sns.set_theme(font_scale=200)


# Function to calculate 95% confidence interval
def confidence_interval(data):
    n = len(data)
    mean = np.mean(data)
    std_err = np.std(data, ddof=1) / np.sqrt(n)
    margin_of_error = 1.96 * std_err
    return mean, mean - margin_of_error, mean + margin_of_error

# Ensure the directory exists
os.makedirs("Results", exist_ok=True)

# Load CIFAR-10 data
def load_cifar10():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Normalize to [0,1]
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Flatten images
    x_train = x_train.reshape(-1, 32 * 32 * 3)
    x_test = x_test.reshape(-1, 32 * 32 * 3)

    # Convert labels to one-hot encoding
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    return (x_train, y_train), (x_test, y_test)

##############################################################################################

def ls_vs_mlp(num_trials=25, epochs=50, batch_size=64):
    (x_train, y_train), (x_test, y_test) = load_cifar10()
    for trial in range(num_trials):
        model, hidden_state_model = create_fc_network(
            input_shape=(32 * 32 * 3,),
            num_reservoir_layers=2,
            layer_base_width=256,
            reservoir_layer_scaling_factor=2,
            num_output_classes=10,
            activation_function="relu",
            frozen=True,
            position="alternating"
        )
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test), verbose=2)
        train_hidden_states = hidden_state_model.predict(x_train)
        test_hidden_states = hidden_state_model.predict(x_test)

        df_ls = evaluate_linear_separability(train_hidden_states, y_train, test_hidden_states, y_test)
        df_mlp = evaluate_linear_separability_mlp(train_hidden_states, y_train, test_hidden_states, y_test)

        train_accuracy = history.history["accuracy"][-1]
        val_accuracy = history.history["val_accuracy"][-1]

        result_dict = {
            "layer_base_width": 256,
            "frozen": True,
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy
        }

        for i in range(len(test_hidden_states)):
            result_dict[f"layer_{i}_train_ls"] = df_ls.loc[i, "Train Linear Separability"]
            result_dict[f"layer_{i}_test_ls"] = df_ls.loc[i, "Test Linear Separability"]
            result_dict[f"layer_{i}_train_mlp"] = df_mlp.loc[i, "Train Linear Separability"]
            result_dict[f"layer_{i}_test_mlp"] = df_mlp.loc[i, "Test Linear Separability"]

        result_df = pd.DataFrame([result_dict])
        with open("Results/cifar_10_solvers.csv", "a") as f:
            result_df.to_csv(f, header=f.tell() == 0, index=False)

# ls_vs_mlp()


##############################################################################################

def visualize_linear_separability(csv_file, base_dataset, data_type="testing"):
    """
    Visualize the average linear separability score and 95% confidence interval for each layer of the network
    based on different types of linear separability solvers.

    Parameters:
    csv_file (str): Path to the CSV file containing the experiment results.
    base_dataset (str): Dataset name.
    data_type (str): One of ["training", "testing"] to determine which scores to visualize.
    """
    if data_type not in ["training", "testing"]:
        raise ValueError("data_type must be one of ['training', 'testing']")

    df = pd.read_csv(csv_file)

    score_column_prefixes = {
        "ls": "layer_{}_train_ls" if data_type == "training" else "layer_{}_test_ls",
        "mlp": "layer_{}_train_mlp" if data_type == "training" else "layer_{}_test_mlp",
    }

    plot_data = []

    for i in range(6):  # Assuming layers 0 to 5 exist
        for score_type, prefix in score_column_prefixes.items():
            layer_column = prefix.format(i)
            if layer_column in df.columns:
                mean, lower, upper = confidence_interval(df[layer_column])
                plot_data.append({
                    "layer": i,
                    "score_type": score_type,
                    "mean_separability": mean,
                    "lower_ci": lower,
                    "upper_ci": upper
                })

    plot_df = pd.DataFrame(plot_data)

    sns.set(style="whitegrid")
    plt.figure(figsize=(7, 5))

    score_type_map = {"ls": "Least Squares", "mlp": "MLP"}
    linestyle_map = {"ls": 'solid', "mlp": 'dashed'}

    # Use the "plasma" colormap with more distinct colors
    colors = sns.color_palette("plasma", n_colors=len(score_type_map))
    # Manually select more distinct colors from the plasma colormap
    distinct_colors = [colors[0], colors[len(colors)//2], colors[-1]]
    color_map = {score_type: distinct_colors[i] for i, score_type in enumerate(score_type_map.keys())}

    for score_type, group in plot_df.groupby('score_type'):
        color = color_map[score_type]
        linestyle = linestyle_map[score_type]
        plt.plot(group["layer"], group["mean_separability"],
                 color=color, linestyle=linestyle, alpha=1, label=score_type_map[score_type])
        plt.fill_between(group["layer"], group["lower_ci"], group["upper_ci"], color=color, alpha=0.1)

    for layer in range(1, 6, 3):
        plt.axvline(x=layer, color='black', linestyle='dotted', linewidth=1.5)

    plt.xticks(np.arange(0, 6, 1))
    plt.grid(True, which='both', linestyle='dotted', linewidth=0.5)

    plt.xlabel("Layer Index")
    plt.ylabel("Average Linear Separability Score")

    # Create legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='lower right', frameon=True, bbox_to_anchor=(1, 0))

    plt.tight_layout()
    filename = f"Results/Images/{base_dataset}_linear_separability_{data_type}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

# visualize_linear_separability("Results/cifar_10_solvers.csv", "cifar_10", "training")
# visualize_linear_separability("Results/cifar_10_solvers.csv", "cifar_10", "testing")


##############################################################################################

def assess_correlation(csv_file, data_type="testing"):
    """
    Assess the correlation between MLP and least squares linear separability scores.

    Parameters:
    csv_file (str): Path to the CSV file containing the experiment results.
    data_type (str): One of ["training", "testing"] to determine which scores to use.
    """
    if data_type not in ["training", "testing"]:
        raise ValueError("data_type must be one of ['training', 'testing']")

    df = pd.read_csv(csv_file)

    score_column_prefixes = {
        "ls": "layer_{}_train_ls" if data_type == "training" else "layer_{}_test_ls",
        "mlp": "layer_{}_train_mlp" if data_type == "training" else "layer_{}_test_mlp"
    }

    ls_scores = []
    mlp_scores = []

    for i in range(6):  # Assuming layers 0 to 5 exist
        ls_column = score_column_prefixes["ls"].format(i)
        mlp_column = score_column_prefixes["mlp"].format(i)

        if ls_column in df.columns and mlp_column in df.columns:
            ls_scores.extend(df[ls_column].dropna())
            mlp_scores.extend(df[mlp_column].dropna())

    # Calculate the correlation coefficient
    correlation_coefficient, _ = pearsonr(ls_scores, mlp_scores)
    print(f"Correlation coefficient between MLP and LS linear separability scores: {correlation_coefficient:.4f}")

# Call the function
assess_correlation("Results/cifar_10_solvers.csv", "training")
assess_correlation("Results/cifar_10_solvers.csv", "testing")