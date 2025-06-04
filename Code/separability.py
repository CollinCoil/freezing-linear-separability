import numpy as np
import tensorflow as tf
import pandas as pd

def train_linear_classifier(X, Y):
    """
    Train a linear classifier using least squares regression.
    
    Parameters:
    X: numpy array of shape (N, F) - N examples with F features each
    Y: numpy array of shape (N, C) - One-hot encoded target classes
    
    Returns:
    W: numpy array of shape (F, C) - Weight matrix mapping features to classes
    """
    W, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    return W

def predict(X, W):
    """
    Make predictions using the trained weights.
    
    Parameters:
    X: numpy array of shape (N, F) - Examples to classify
    W: numpy array of shape (F, C) - Trained weight matrix
    
    Returns:
    predictions: numpy array of shape (N,) - Predicted class indices
    """
    scores = X @ W
    return np.eye(W.shape[1])[np.argmax(scores, axis=1)]

def evaluate_linear_separability(train_hidden_states, train_labels, test_hidden_states, test_labels):
    """
    Evaluate linear separability at each hidden layer using separate training and testing sets.
    
    Parameters:
    train_hidden_states: list of numpy arrays - Feature representations at each hidden layer (training set)
    train_labels: numpy array - True labels for training set (one-hot encoded)
    test_hidden_states: list of numpy arrays - Feature representations at each hidden layer (testing set)
    test_labels: numpy array - True labels for testing set (one-hot encoded)
    
    Returns:
    pandas DataFrame with layer indices and corresponding training and testing accuracy scores.
    """
    results = []
    
    # Convert one-hot encoded labels to class indices
    train_label_indices = np.argmax(train_labels, axis=1)
    test_label_indices = np.argmax(test_labels, axis=1)
    
    for i, (train_features, test_features) in enumerate(zip(train_hidden_states, test_hidden_states)):
        W = train_linear_classifier(train_features, train_labels)  # Train on training data
        
        # Get predictions as one-hot encodings
        train_predictions = predict(train_features, W)
        test_predictions = predict(test_features, W)
        
        # Convert predictions to class indices
        train_pred_indices = np.argmax(train_predictions, axis=1)
        test_pred_indices = np.argmax(test_predictions, axis=1)
        
        # Calculate accuracy by comparing class indices
        train_accuracy = np.mean(train_pred_indices == train_label_indices)
        test_accuracy = np.mean(test_pred_indices == test_label_indices)
        
        results.append({"Layer Index": i, 
                        "Train Linear Separability": train_accuracy, 
                        "Test Linear Separability": test_accuracy})
    
    return pd.DataFrame(results)


def evaluate_linear_separability_mlp(train_hidden_states, train_labels, test_hidden_states, test_labels, epochs=10, batch_size=128):
    """
    Evaluate linear separability using a one-layer MLP (Dense + softmax).
    
    Returns:
    pandas DataFrame with layer indices and training/testing accuracy.
    """
    results = []

    train_label_indices = np.argmax(train_labels, axis=1)
    test_label_indices = np.argmax(test_labels, axis=1)
    num_classes = train_labels.shape[1]

    for i, (train_features, test_features) in enumerate(zip(train_hidden_states, test_hidden_states)):
        input_dim = train_features.shape[1]

        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(train_features, train_labels, epochs=epochs, batch_size=batch_size, verbose=0)

        train_accuracy = model.evaluate(train_features, train_labels, verbose=0)[1]
        test_accuracy = model.evaluate(test_features, test_labels, verbose=0)[1]

        results.append({
            "Layer Index": i,
            "Train Linear Separability": train_accuracy,
            "Test Linear Separability": test_accuracy
        })

    return pd.DataFrame(results)