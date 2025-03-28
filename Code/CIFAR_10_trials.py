import os
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from create_network import create_fc_network, create_fc_network_frozen_weights  # Import from create_network.py
from separability import evaluate_linear_separability  # Import from separability.py

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

def evaluate_width(layer_base_widths, frozen_options, num_trials=25, epochs=100, batch_size=64):
    # Load data
    (x_train, y_train), (x_test, y_test) = load_cifar10()

    # Results DataFrame
    results = []

    # Run experiments
    for layer_base_width in layer_base_widths:
        for frozen in frozen_options:
            trial_results = []
            for trial in range(num_trials):
                # Initialize network
                model, hidden_state_model = create_fc_network(
                    input_shape=(32 * 32 * 3,),
                    num_reservoir_layers=2,
                    layer_base_width=layer_base_width,
                    reservoir_layer_scaling_factor=2,
                    num_output_classes=10,
                    activation_function="relu",
                    frozen=frozen,
                    position="alternating"
                )

                # Compile the model
                model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

                # Train the model on CIFAR-10
                history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test), verbose=0)

                # Extract hidden states for training and test sets
                train_hidden_states = hidden_state_model.predict(x_train)
                test_hidden_states = hidden_state_model.predict(x_test)

                # Evaluate linear separability using updated function
                df_results = evaluate_linear_separability(train_hidden_states, y_train, test_hidden_states, y_test)

                # Get the accuracy from the training history
                train_accuracy = history.history["accuracy"][-1]
                val_accuracy = history.history["val_accuracy"][-1]

                # Store separability scores for each layer
                train_separability_scores = [df_results.loc[i, "Train Linear Separability"] for i in range(len(test_hidden_states))]
                test_separability_scores = [df_results.loc[i, "Test Linear Separability"] for i in range(len(test_hidden_states))]

                # Create a result dictionary for this combination of layer_base_width and frozen
                result_dict = {
                    "layer_base_width": layer_base_width,
                    "frozen": frozen,
                    "train_accuracy": train_accuracy,
                    "val_accuracy": val_accuracy
                }
                for i, (train_score, test_score) in enumerate(zip(train_separability_scores, test_separability_scores)):
                    result_dict[f"layer_{i}_train_separability"] = train_score
                    result_dict[f"layer_{i}_test_separability"] = test_score

                # Convert the result dictionary to a DataFrame
                result_df = pd.DataFrame([result_dict])

                # Append the result to the CSV file
                with open("Results/cifar_10_width.csv", "a") as f:
                    result_df.to_csv(f, header=f.tell() == 0, index=False)

# Example usage
layer_base_widths = [32, 64, 128, 256, 512, 1024]
frozen_options = [True, False]
evaluate_width(layer_base_widths, frozen_options)


##############################################################################################

def evaluate_depth(num_reservoir_layers, frozen_options, num_trials=25, epochs=100, batch_size=64):
    # Load data
    (x_train, y_train), (x_test, y_test) = load_cifar10()

    # Results DataFrame
    results = []

    # Run experiments
    for num_layers in num_reservoir_layers:
        for frozen in frozen_options:
            trial_results = []
            for trial in range(num_trials):
                # Initialize network
                model, hidden_state_model = create_fc_network(
                    input_shape=(32 * 32 * 3,),
                    num_reservoir_layers=num_layers,
                    layer_base_width=256,
                    reservoir_layer_scaling_factor=2,
                    num_output_classes=10,
                    activation_function="relu",
                    frozen=frozen,
                    position="alternating"
                )

                # Compile the model
                model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

                # Train the model on CIFAR-10
                history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test), verbose=0)

                # Extract hidden states for training and test sets
                train_hidden_states = hidden_state_model.predict(x_train)
                test_hidden_states = hidden_state_model.predict(x_test)

                # Evaluate linear separability using updated function
                df_results = evaluate_linear_separability(train_hidden_states, y_train, test_hidden_states, y_test)

                # Get the accuracy from the training history
                train_accuracy = history.history["accuracy"][-1]
                val_accuracy = history.history["val_accuracy"][-1]

                # Store separability scores for each layer
                train_separability_scores = [df_results.loc[i, "Train Linear Separability"] for i in range(len(test_hidden_states))]
                test_separability_scores = [df_results.loc[i, "Test Linear Separability"] for i in range(len(test_hidden_states))]

                # Create a result dictionary for this combination of num_layers and frozen
                result_dict = {
                    "num_reservoir_layers": num_layers,
                    "frozen": frozen,
                    "train_accuracy": train_accuracy,
                    "val_accuracy": val_accuracy
                }
                for i, (train_score, test_score) in enumerate(zip(train_separability_scores, test_separability_scores)):
                    result_dict[f"layer_{i}_train_separability"] = train_score
                    result_dict[f"layer_{i}_test_separability"] = test_score

                # Convert the result dictionary to a DataFrame
                result_df = pd.DataFrame([result_dict])

                # Append the result to the CSV file
                with open("Results/cifar_10_depth.csv", "a") as f:
                    result_df.to_csv(f, header=f.tell() == 0, index=False)


# Example usage
num_reservoir_layers = [8, 7, 6, 5, 4, 3, 2, 1]
frozen_options = [True, False]
evaluate_depth(num_reservoir_layers, frozen_options)

##############################################################################################

def evaluate_depth_with_skip(num_reservoir_layers, frozen_options, use_skip_connections_options, num_trials=25, epochs=100, batch_size=64):
    # Load data
    (x_train, y_train), (x_test, y_test) = load_cifar10()

    # Results DataFrame
    results = []

    # Run experiments
    for num_layers in num_reservoir_layers:
        for frozen in frozen_options:
            for use_skip_connections in use_skip_connections_options:
                trial_results = []
                for trial in range(num_trials):
                    # Initialize network
                    model, hidden_state_model = create_fc_network(
                        input_shape=(32 * 32 * 3,),
                        num_reservoir_layers=num_layers,
                        layer_base_width=256,
                        reservoir_layer_scaling_factor=2,
                        num_output_classes=10,
                        activation_function="relu",
                        frozen=frozen,
                        position="alternating",
                        use_skip_connections=use_skip_connections  # New option
                    )

                    # Compile the model
                    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

                    # Train the model on CIFAR-10
                    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test), verbose=0)

                    # Extract hidden states for training and test sets
                    train_hidden_states = hidden_state_model.predict(x_train)
                    test_hidden_states = hidden_state_model.predict(x_test)

                    # Evaluate linear separability
                    df_results = evaluate_linear_separability(train_hidden_states, y_train, test_hidden_states, y_test)

                    # Get the accuracy from the training history
                    train_accuracy = history.history["accuracy"][-1]
                    val_accuracy = history.history["val_accuracy"][-1]

                    # Store separability scores for each layer
                    train_separability_scores = [df_results.loc[i, "Train Linear Separability"] for i in range(len(test_hidden_states))]
                    test_separability_scores = [df_results.loc[i, "Test Linear Separability"] for i in range(len(test_hidden_states))]

                    # Create a result dictionary for this combination of num_layers, frozen, and skip connections
                    result_dict = {
                        "num_reservoir_layers": num_layers,
                        "frozen": frozen,
                        "use_skip_connections": use_skip_connections,
                        "train_accuracy": train_accuracy,
                        "val_accuracy": val_accuracy
                    }
                    for i, (train_score, test_score) in enumerate(zip(train_separability_scores, test_separability_scores)):
                        result_dict[f"layer_{i}_train_separability"] = train_score
                        result_dict[f"layer_{i}_test_separability"] = test_score

                    # Convert the result dictionary to a DataFrame
                    result_df = pd.DataFrame([result_dict])

                    # Append the result to the CSV file
                    with open("Results/cifar_100_depth_skip.csv", "a") as f:
                        result_df.to_csv(f, header=f.tell() == 0, index=False)


# Example usage with skip connections enabled/disabled
num_reservoir_layers = [8, 7, 6, 5, 4, 3, 2]
frozen_options = [True, False]
use_skip_connections_options = [True]  # New skip connection flag

evaluate_depth_with_skip(num_reservoir_layers, frozen_options, use_skip_connections_options)

##############################################################################################

def evaluate_reservoir_scaling(reservoir_layer_scaling_factors, frozen_options, num_trials=25, epochs=100, batch_size=64):
    # Load data
    (x_train, y_train), (x_test, y_test) = load_cifar10()

    # Results DataFrame
    results = []

    # Run experiments
    for scaling_factor in reservoir_layer_scaling_factors:
        for frozen in frozen_options:
            trial_results = []
            for trial in range(num_trials):
                # Initialize network
                model, hidden_state_model = create_fc_network(
                    input_shape=(32 * 32 * 3,),
                    num_reservoir_layers=2,
                    layer_base_width=256,
                    reservoir_layer_scaling_factor=scaling_factor,
                    num_output_classes=10,
                    activation_function="relu",
                    frozen=frozen,
                    position="alternating"
                )

                # Compile the model
                model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

                # Train the model on CIFAR-10
                history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test), verbose=0)

                # Extract hidden states for training and test sets
                train_hidden_states = hidden_state_model.predict(x_train)
                test_hidden_states = hidden_state_model.predict(x_test)

                # Evaluate linear separability using updated function
                df_results = evaluate_linear_separability(train_hidden_states, y_train, test_hidden_states, y_test)

                # Get the accuracy from the training history
                train_accuracy = history.history["accuracy"][-1]
                val_accuracy = history.history["val_accuracy"][-1]

                # Store separability scores for each layer
                train_separability_scores = [df_results.loc[i, "Train Linear Separability"] for i in range(len(test_hidden_states))]
                test_separability_scores = [df_results.loc[i, "Test Linear Separability"] for i in range(len(test_hidden_states))]

                # Create a result dictionary for this combination of scaling_factor and frozen
                result_dict = {
                    "reservoir_layer_scaling_factor": scaling_factor,
                    "frozen": frozen,
                    "train_accuracy": train_accuracy,
                    "val_accuracy": val_accuracy
                }
                for i, (train_score, test_score) in enumerate(zip(train_separability_scores, test_separability_scores)):
                    result_dict[f"layer_{i}_train_separability"] = train_score
                    result_dict[f"layer_{i}_test_separability"] = test_score

                # Convert the result dictionary to a DataFrame
                result_df = pd.DataFrame([result_dict])

                # Append the result to the CSV file
                with open("Results/cifar_10_scaling.csv", "a") as f:
                    result_df.to_csv(f, header=f.tell() == 0, index=False)


# Example usage
reservoir_layer_scaling_factors = [0.25, 0.5, 1, 2, 4, 8, 16, 32]
frozen_options = [True, False]
evaluate_reservoir_scaling(reservoir_layer_scaling_factors, frozen_options)

##############################################################################################

def evaluate_frozen_position(positions, frozen_options, num_trials=25, epochs=100, batch_size=64):
    # Load data
    (x_train, y_train), (x_test, y_test) = load_cifar10()

    # Results DataFrame
    results = []

    # Run experiments
    for position in positions:
        for frozen in frozen_options:
            trial_results = []
            for trial in range(num_trials):
                # Initialize network
                model, hidden_state_model = create_fc_network(
                    input_shape=(32 * 32 * 3,),
                    num_reservoir_layers=2,
                    layer_base_width=256,
                    reservoir_layer_scaling_factor=2,
                    num_output_classes=10,
                    activation_function="relu",
                    frozen=frozen,
                    position=position
                )

                # Compile the model
                model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

                # Train the model on CIFAR-10
                history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test), verbose=0)

                # Extract hidden states for training and test sets
                train_hidden_states = hidden_state_model.predict(x_train)
                test_hidden_states = hidden_state_model.predict(x_test)

                # Evaluate linear separability using updated function
                df_results = evaluate_linear_separability(train_hidden_states, y_train, test_hidden_states, y_test)

                # Get the accuracy from the training history
                train_accuracy = history.history["accuracy"][-1]
                val_accuracy = history.history["val_accuracy"][-1]

                # Store separability scores for each layer
                train_separability_scores = [df_results.loc[i, "Train Linear Separability"] for i in range(len(test_hidden_states))]
                test_separability_scores = [df_results.loc[i, "Test Linear Separability"] for i in range(len(test_hidden_states))]

                # Create a result dictionary for this combination of position and frozen
                result_dict = {
                    "position": position,
                    "frozen": frozen,
                    "train_accuracy": train_accuracy,
                    "val_accuracy": val_accuracy
                }
                for i, (train_score, test_score) in enumerate(zip(train_separability_scores, test_separability_scores)):
                    result_dict[f"layer_{i}_train_separability"] = train_score
                    result_dict[f"layer_{i}_test_separability"] = test_score

                # Convert the result dictionary to a DataFrame
                result_df = pd.DataFrame([result_dict])

                # Append the result to the CSV file
                with open("Results/cifar_10_position.csv", "a") as f:
                    result_df.to_csv(f, header=f.tell() == 0, index=False)

# Example usage
positions = ["alternating", "front", "middle", "back"]
frozen_options = [True, False]
evaluate_frozen_position(positions, frozen_options)

##############################################################################################


def evaluate_regularization(regularization_options, frozen_options, num_trials=25, epochs=100, batch_size=64):
    (x_train, y_train), (x_test, y_test) = load_cifar10()
    results = []
    
    for reg_name, reg_params in regularization_options:
        for frozen in frozen_options:
            for trial in range(num_trials):
                model, hidden_state_model = create_fc_network(
                    input_shape=(32 * 32 * 3,),
                    num_reservoir_layers=2,
                    layer_base_width=256,
                    reservoir_layer_scaling_factor=2,
                    num_output_classes=10,
                    activation_function="relu",
                    frozen=frozen,
                    position="alternating",
                    **reg_params
                )
                
                model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
                history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test), verbose=0)
                
                train_hidden_states = hidden_state_model.predict(x_train)
                test_hidden_states = hidden_state_model.predict(x_test)
                df_results = evaluate_linear_separability(train_hidden_states, y_train, test_hidden_states, y_test)
                
                train_accuracy = history.history["accuracy"][-1]
                val_accuracy = history.history["val_accuracy"][-1]
                
                train_separability_scores = [df_results.loc[i, "Train Linear Separability"] for i in range(len(test_hidden_states))]
                test_separability_scores = [df_results.loc[i, "Test Linear Separability"] for i in range(len(test_hidden_states))]
                
                result_dict = {
                    "regularization": reg_name,
                    "frozen": frozen,
                    "train_accuracy": train_accuracy,
                    "val_accuracy": val_accuracy
                }
                for i, (train_score, test_score) in enumerate(zip(train_separability_scores, test_separability_scores)):
                    result_dict[f"layer_{i}_train_separability"] = train_score
                    result_dict[f"layer_{i}_test_separability"] = test_score
                
                result_df = pd.DataFrame([result_dict])
                with open("Results/cifar_10_regularization.csv", "a") as f:
                    result_df.to_csv(f, header=f.tell() == 0, index=False)

# Regularization strategies
regularization_options = [
    ("l1_0.001", {"use_l1": True, "l1_lambda": 0.001}),
    ("l1_0.01", {"use_l1": True, "l1_lambda": 0.01}),
    ("batchnorm", {"use_batchnorm": True}),
    ("dropout_0.1", {"use_dropout": True, "dropout_rate": 0.1}),
    ("dropout_0.25", {"use_dropout": True, "dropout_rate": 0.25}),
    ("dropout_0.5", {"use_dropout": True, "dropout_rate": 0.5})
]
frozen_options = [True, False]
evaluate_regularization(regularization_options, frozen_options)

##############################################################################################

def evaluate_training_progress(frozen_options, num_trials=25, epochs=100, batch_size=64):
    # Load data
    (x_train, y_train), (x_test, y_test) = load_cifar10()

    # Results DataFrame
    results = []

    # Run experiments
    for frozen in frozen_options:
        for trial in range(num_trials):
            # Initialize network
            model, hidden_state_model = create_fc_network(
                input_shape=(32 * 32 * 3,),
                num_reservoir_layers=2,
                layer_base_width=256,
                reservoir_layer_scaling_factor=2,
                num_output_classes=10,
                activation_function="relu",
                frozen=frozen,  # Set frozen option
                position="alternating"
            )

            # Compile the model
            model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            
            # Evaluate separability and zero-shot accuracy before training (Epoch 0)
            train_hidden_states = hidden_state_model.predict(x_train)
            test_hidden_states = hidden_state_model.predict(x_test)
            df_results = evaluate_linear_separability(train_hidden_states, y_train, test_hidden_states, y_test)
            
            train_loss, train_accuracy = model.evaluate(x_train, y_train, verbose=0)
            val_loss, val_accuracy = model.evaluate(x_test, y_test, verbose=0)
            
            initial_result_dict = {"trial": trial, "epoch": 0, "frozen": frozen, "train_accuracy": train_accuracy, "val_accuracy": val_accuracy}
            for i in range(len(test_hidden_states)):
                initial_result_dict[f"layer_{i}_train_separability"] = df_results.loc[i, "Train Linear Separability"]
                initial_result_dict[f"layer_{i}_test_separability"] = df_results.loc[i, "Test Linear Separability"]
            
            initial_result_df = pd.DataFrame([initial_result_dict])
            with open("Results/cifar_10_training_separability.csv", "a") as f:
                initial_result_df.to_csv(f, header=f.tell() == 0, index=False)
            
            # Train model and evaluate each epoch
            for epoch in range(epochs):
                history = model.fit(x_train, y_train, epochs=1, batch_size=batch_size, validation_data=(x_test, y_test), verbose=0)
                
                # Extract hidden states for training and test sets
                train_hidden_states = hidden_state_model.predict(x_train)
                test_hidden_states = hidden_state_model.predict(x_test)

                # Evaluate linear separability
                df_results = evaluate_linear_separability(train_hidden_states, y_train, test_hidden_states, y_test)
                
                # Get the accuracy from the training history
                train_accuracy = history.history["accuracy"][-1]
                val_accuracy = history.history["val_accuracy"][-1]

                # Create a result dictionary for this epoch
                result_dict = {
                    "trial": trial, 
                    "epoch": epoch + 1, 
                    "frozen": frozen,
                    "train_accuracy": train_accuracy, 
                    "val_accuracy": val_accuracy
                }
                
                for i in range(len(test_hidden_states)):
                    result_dict[f"layer_{i}_train_separability"] = df_results.loc[i, "Train Linear Separability"]
                    result_dict[f"layer_{i}_test_separability"] = df_results.loc[i, "Test Linear Separability"]
                
                # Convert the result dictionary to a Dataframe
                result_df = pd.DataFrame([result_dict])

                # Append the result to the CSV file
                with open("Results/cifar_10_training_separability.csv", "a") as f:
                    result_df.to_csv(f, header=f.tell() == 0, index=False)


# Example usage
frozen_options = [True, False]
evaluate_training_progress(frozen_options)

##############################################################################################

def evaluate_training_progress_frozen_weights(freeze_ratios, num_trials=25, epochs=100, batch_size=64):
    # Load data
    (x_train, y_train), (x_test, y_test) = load_cifar10()
    
    # Results list
    results = []
    
    # Run experiments
    for freeze_ratio in freeze_ratios:
        for trial in range(num_trials):
            # Initialize network with partial weight freezing
            model, hidden_state_model = create_fc_network_frozen_weights(
                input_shape=(32 * 32 * 3,),
                num_reservoir_layers=2,
                layer_base_width=256,
                reservoir_layer_scaling_factor=2,
                num_output_classes=10,
                activation_function="relu",
                freeze_ratio=freeze_ratio  # Use freeze_ratio instead of frozen
            )
            
            # Compile the model
            model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            
            # Evaluate separability and zero-shot accuracy before training (Epoch 0)
            train_hidden_states = hidden_state_model.predict(x_train)
            test_hidden_states = hidden_state_model.predict(x_test)
            df_results = evaluate_linear_separability(train_hidden_states, y_train, test_hidden_states, y_test)
            
            train_loss, train_accuracy = model.evaluate(x_train, y_train, verbose=0)
            val_loss, val_accuracy = model.evaluate(x_test, y_test, verbose=0)
            
            initial_result_dict = {
                "trial": trial, 
                "epoch": 0, 
                "freeze_ratio": freeze_ratio,  # Record freeze_ratio instead of frozen
                "train_accuracy": train_accuracy, 
                "val_accuracy": val_accuracy
            }
            
            for i in range(len(train_hidden_states)):
                initial_result_dict[f"layer_{i}_train_separability"] = df_results.loc[i, "Train Linear Separability"]
                initial_result_dict[f"layer_{i}_test_separability"] = df_results.loc[i, "Test Linear Separability"]
            
            initial_result_df = pd.DataFrame([initial_result_dict])
            with open("Results/cifar_10_partial_freezing_separability.csv", "a") as f:
                initial_result_df.to_csv(f, header=f.tell() == 0, index=False)
            
            # Train model and evaluate each epoch
            for epoch in range(epochs):
                history = model.fit(x_train, y_train, epochs=1, batch_size=batch_size, validation_data=(x_test, y_test), verbose=0)
                
                # Extract hidden states for training and test sets
                train_hidden_states = hidden_state_model.predict(x_train)
                test_hidden_states = hidden_state_model.predict(x_test)
                
                # Evaluate linear separability
                df_results = evaluate_linear_separability(train_hidden_states, y_train, test_hidden_states, y_test)
                
                # Get the accuracy from the training history
                train_accuracy = history.history["accuracy"][-1]
                val_accuracy = history.history["val_accuracy"][-1]
                
                # Create a result dictionary for this epoch
                result_dict = {
                    "trial": trial, 
                    "epoch": epoch + 1, 
                    "freeze_ratio": freeze_ratio,  # Record freeze_ratio instead of frozen
                    "train_accuracy": train_accuracy, 
                    "val_accuracy": val_accuracy
                }
                
                for i in range(len(train_hidden_states)):
                    result_dict[f"layer_{i}_train_separability"] = df_results.loc[i, "Train Linear Separability"]
                    result_dict[f"layer_{i}_test_separability"] = df_results.loc[i, "Test Linear Separability"]
                
                # Convert the result dictionary to a Dataframe
                result_df = pd.DataFrame([result_dict])
                
                # Append the result to the CSV file
                with open("Results/cifar_10_partial_freezing_separability.csv", "a") as f:
                    result_df.to_csv(f, header=f.tell() == 0, index=False)

# Example usage with different freeze ratios
freeze_ratios = [0.1, 0.2, 0.3, 0.4, 0.5]
evaluate_training_progress_frozen_weights(freeze_ratios)