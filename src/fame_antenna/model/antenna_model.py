import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
K.set_floatx('float32')

def build_factorise_model(scalers, n_antenna, n_maps, n_s, num_jets, loop_scale):
    p_input = keras.Input(shape=(num_jets, 4,), name="ps")
    map_input = keras.Input(shape=(n_maps,), name="recoil")
    mu_input = keras.Input(shape=(1,), name="mu_r")
    s_input = keras.Input(shape=(n_s,), name="sij")
    antenna = keras.Input(shape=(n_antenna,), name="antenna")
    born = keras.Input(shape=(1,), name="born")
    born_weight = keras.Input(shape=(1,), name="born_weight")
    loop = keras.Input(shape=(1,), name="loop")
    targets = keras.Input(shape=(1,), name="me")
    
    inputs = [p_input, map_input, mu_input, s_input, antenna, targets, born, born_weight, loop]
    
    act_func = "swish"
    kernel = "glorot_uniform"
    
    y_scaler, y_scaler_loop, mom_scaler, map_scaler, mu_scaler, sij_scaler = scalers

    x = mom_scaler(p_input)
    x = layers.Flatten()(x)
    m = map_scaler(map_input)
    mu = mu_scaler(mu_input)
    s = sij_scaler(s_input)
    x = layers.Concatenate()([x, m, mu, s])
    x = layers.Dense(64, activation=act_func, kernel_initializer=kernel)(x)
    x = layers.Dense(64, activation=act_func, kernel_initializer=kernel)(x)
    x = layers.Dense(64, activation=act_func, kernel_initializer=kernel)(x)
    outputs = layers.Dense(n_antenna + 1, activation=act_func)(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name=f"{num_jets}_jets")
    model.add_loss(custom_loss(targets, outputs, antenna, born_weight, y_scaler))
    model.add_metric(k_loss(targets, outputs, born_weight, antenna, y_scaler), name="k")
    model.add_metric(loop_loss(born, loop, outputs, antenna, y_scaler_loop, loop_scale), name="l")
    adam = keras.optimizers.Adam(learning_rate=1E-3)
    model.compile(loss=None, optimizer=adam)
    return model

def build_bare_factorise_model(scalers, n_antenna, n_maps, n_s, num_jets):
    p_input = keras.Input(shape=(num_jets, 4,), name="ps")
    map_input = keras.Input(shape=(n_maps,), name="recoil")
    mu_input = keras.Input(shape=(1,), name="mu_r")
    s_input = keras.Input(shape=(n_s,), name="sij")
    
    inputs = [p_input, map_input, mu_input, s_input]
    
    act_func = "swish"
    kernel = "glorot_uniform"

    _, mom_scaler, map_scaler, mu_scaler, sij_scaler = scalers
    
    x = mom_scaler(p_input)
    x = layers.Flatten()(x)
    m = map_scaler(map_input)
    mu = mu_scaler(mu_input)
    s = sij_scaler(s_input)
    x = layers.Concatenate()([x, m, mu, s])
    x = layers.Dense(64, activation=act_func, kernel_initializer=kernel)(x)
    x = layers.Dense(64, activation=act_func, kernel_initializer=kernel)(x)
    x = layers.Dense(64, activation=act_func, kernel_initializer=kernel)(x)
    outputs = layers.Dense(n_antenna + 1, activation=act_func)(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name=f"{num_jets}_jets")
    return model

def custom_loss(y_true, y_pred, antenna, born_weight, y_scaler):
    loss_k = k_loss(y_true, y_pred, born_weight, antenna, y_scaler)
    loss = loss_k
    return loss

def loop_loss(born, loop, y_pred, antenna, y_scaler_loop, loop_scale):
    k_preds = y_pred[:, 0] + tf.math.reduce_sum(y_pred[:, 1:]*antenna, axis=1)
    y_preds = k_preds[:, None]*born
    y_preds = y_scaler_loop(tf.math.asinh(y_preds/loop_scale))
    loop_loss = keras.losses.mean_squared_error(loop, y_preds)
    return loop_loss

def k_loss(y_true, y_pred, born_weight, antenna, y_scaler):
    y_preds = y_pred[:, 0] + tf.math.reduce_sum(y_pred[:, 1:]*antenna, axis=1)
    y_preds = y_scaler(y_preds[:, None])
    k_loss = keras.losses.mean_absolute_error(
        tf.math.multiply(y_true, born_weight),
        tf.math.multiply(y_preds, born_weight)
    )
    return k_loss