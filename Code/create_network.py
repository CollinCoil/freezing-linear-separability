import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import numpy as np

def create_fc_network(input_shape, num_reservoir_layers, layer_base_width, 
                      reservoir_layer_scaling_factor, num_output_classes, 
                      activation_function, frozen, position,
                      use_l1=False, l1_lambda=0.01,
                      use_batchnorm=False, use_dropout=False, dropout_rate=0.5,
                      use_skip_connections=False):
    """
    Creates a fully connected neural network with configurable reservoir layers.
    Allows extraction of hidden state activations and optional regularization.
    Supports skip connections for alternating networks when enabled.
    """
    inputs = layers.Input(shape=input_shape)
    x = inputs

    reservoir_width = int(layer_base_width * reservoir_layer_scaling_factor)
    num_trainable_layers = 2 * num_reservoir_layers
    hidden_layers = []

    skip_input = None  # Stores the first base layer output from the previous block

    def add_layer(units, is_reservoir, use_skip=False):
        nonlocal x, skip_input
        kernel_regularizer = regularizers.L1(l1_lambda) if use_l1 else None
        
        layer = layers.Dense(units, activation=activation_function, 
                             kernel_regularizer=kernel_regularizer, 
                             name=f"hidden_{len(hidden_layers)}")
        
        new_x = layer(x)

        # Apply skip connection if enabled and if not the first block
        if use_skip and use_skip_connections and skip_input is not None:
            new_x = layers.Add()([new_x, skip_input])  # Add the skip connection
        
        if use_batchnorm:
            new_x = layers.BatchNormalization()(new_x)
        
        if use_dropout:
            new_x = layers.Dropout(dropout_rate)(new_x)
        
        hidden_layers.append(new_x)
        x = new_x  # Update x for next layer
        
        if is_reservoir and frozen:
            layer.trainable = False

        return new_x

    if position == "alternating":
        for i in range(num_reservoir_layers):
            # First base layer (trainable) - potential skip source
            first_base_layer_output = add_layer(layer_base_width, is_reservoir=False, use_skip=(i > 0))
            
            # Reservoir layer (frozen)
            add_layer(reservoir_width, is_reservoir=True)
            
            # Second base layer (trainable) - normal, no skip
            add_layer(layer_base_width, is_reservoir=False)

            # Store the first base layer output for the next block's skip connection
            skip_input = first_base_layer_output if use_skip_connections else None

    elif position == "front":
        for _ in range(num_reservoir_layers):
            add_layer(reservoir_width, is_reservoir=True)
        for _ in range(num_trainable_layers):
            add_layer(layer_base_width, is_reservoir=False)
    
    elif position == "middle":
        for _ in range(num_reservoir_layers):
            add_layer(layer_base_width, is_reservoir=False)
        for _ in range(num_reservoir_layers):
            add_layer(reservoir_width, is_reservoir=True)
        for _ in range(num_reservoir_layers):
            add_layer(layer_base_width, is_reservoir=False)
    
    elif position == "back":
        for _ in range(num_trainable_layers):
            add_layer(layer_base_width, is_reservoir=False)
        for _ in range(num_reservoir_layers):
            add_layer(reservoir_width, is_reservoir=True)

    outputs = layers.Dense(num_output_classes, activation="softmax", name="output")(x)

    # Create main model
    model = models.Model(inputs=inputs, outputs=outputs)

    # Create a model that outputs hidden states
    hidden_state_model = models.Model(inputs=inputs, outputs=hidden_layers)

    return model, hidden_state_model


class PartiallyFrozenDense(layers.Layer):
    def __init__(self, units, activation=None, freeze_mask=None, **kwargs):
        super(PartiallyFrozenDense, self).__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.freeze_mask = freeze_mask  # Boolean mask where True indicates frozen weights
        
    def build(self, input_shape):
        input_dim = input_shape[-1]
        
        # Initialize weights and bias - fix here by removing 'shape' from kwargs
        self.kernel = self.add_weight(
            name='kernel',
            shape=[input_dim, self.units],
            initializer='glorot_uniform',
            trainable=True)
        
        self.bias = self.add_weight(
            name='bias',
            shape=[self.units,],
            initializer='zeros',
            trainable=True)
            
        # If no mask provided, create one with all weights trainable
        if self.freeze_mask is None:
            self.freeze_mask = tf.zeros_like(self.kernel, dtype=bool)
        else:
            # Ensure mask has correct shape
            self.freeze_mask = tf.cast(tf.reshape(self.freeze_mask, self.kernel.shape), dtype=bool)
        
        super(PartiallyFrozenDense, self).build(input_shape)
        
    def call(self, inputs):
        # Apply frozen mask using custom gradient
        @tf.custom_gradient
        def masked_kernel(kernel):
            def grad(upstream):
                # Only propagate gradients where mask is False (unfrozen weights)
                return upstream * tf.cast(~self.freeze_mask, dtype=tf.float32)
            return kernel, grad
            
        masked_weights = masked_kernel(self.kernel)
        outputs = tf.matmul(inputs, masked_weights)
        outputs = tf.nn.bias_add(outputs, self.bias)
        
        if self.activation is not None:
            outputs = self.activation(outputs)
            
        return outputs

def create_fc_network_frozen_weights(input_shape, num_reservoir_layers, layer_base_width, 
                      reservoir_layer_scaling_factor, num_output_classes, 
                      activation_function, freeze_ratio=0.8):
    """
    Creates a fully connected neural network with alternating base and reservoir layers.
    Reservoir layers have individual weights frozen according to freeze_ratio.
    """
    inputs = layers.Input(shape=input_shape)
    x = inputs
    reservoir_width = int(layer_base_width * reservoir_layer_scaling_factor)
    hidden_layers = []
    
    def add_layer(units, is_reservoir):
        nonlocal x
        
        if is_reservoir:
            # Create random freeze mask where freeze_ratio of weights are frozen
            freeze_mask = tf.random.uniform(
                shape=[x.shape[-1], units], 
                minval=0, 
                maxval=1) < freeze_ratio
                
            layer = PartiallyFrozenDense(
                units=units,
                activation=activation_function,
                freeze_mask=freeze_mask,
                name=f"hidden_{len(hidden_layers)}"
            )
        else:
            # Regular trainable layer
            layer = layers.Dense(
                units=units, 
                activation=activation_function,
                name=f"hidden_{len(hidden_layers)}"
            )
        
        x = layer(x)
        hidden_layers.append(x)
            
    # Always using alternating pattern
    for i in range(num_reservoir_layers):
        # First base layer (trainable)
        add_layer(layer_base_width, is_reservoir=False)
        
        # Reservoir layer (partially frozen)
        add_layer(reservoir_width, is_reservoir=True)
        
        # Second base layer (trainable)
        add_layer(layer_base_width, is_reservoir=False)
    
    outputs = layers.Dense(num_output_classes, activation="softmax", name="output")(x)
    
    # Create main model
    model = models.Model(inputs=inputs, outputs=outputs)
    
    # Create a model that outputs hidden states
    hidden_state_model = models.Model(inputs=inputs, outputs=hidden_layers)
    
    return model, hidden_state_model