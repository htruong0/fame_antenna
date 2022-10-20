import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
K.set_floatx('float32')

def build_base_model(scalers, num_jets):
    p_input = keras.Input(shape=(num_jets, 4,))
    mu_input = keras.Input(shape=(1,))
    inputs = [p_input, mu_input]
    
    act_func = "swish"
    kernel = "glorot_uniform"

    mom_scaler, mu_scaler = scalers
    x = mom_scaler(p_input)
    mu = mu_scaler(mu_input)
    x = layers.Flatten()(x)
    x = layers.Concatenate()([x, mu])
    x = layers.Dense(64, activation=act_func, kernel_initializer=kernel)(x)
    x = layers.Dense(64, activation=act_func, kernel_initializer=kernel)(x)
    x = layers.Dense(64, activation=act_func, kernel_initializer=kernel)(x)
    outputs = layers.Dense(1)(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name=f"{num_jets}_jets")
    return model