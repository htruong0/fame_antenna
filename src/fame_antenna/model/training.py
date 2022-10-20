import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from fame_antenna.model.custom_callbacks import RatioEarlyStopping

def train_model(
        model,
        X,
        Y,
        checkpoint_path,
        epochs=10000,
        batch_size=512,
        learning_rate=1E-3,
        min_delta=1E-6,
        loss=None,
        reduce_lr=True,
        force_save=False
    ):
    '''Wrapper function to train model with useful monitoring tools and saving models.'''

    # since model has custom loss added can pass loss=None here
    model.compile(
        loss=loss,
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate)
    )

    # training termination critera, min_delta is tuned such that models don't train forever
    es = keras.callbacks.EarlyStopping(
        monitor='val_l',
        mode='min',
        patience=100,
        verbose=1,
        min_delta=min_delta,
        restore_best_weights=True
    )
    ratio = RatioEarlyStopping(
        ratio=0.9,
        patience=40,
        verbose=1,
        restore_best_weights=True
    )

    callbacks = [es, ratio]
    # learning rate reduction helps convergence
    if reduce_lr:
        lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_l',
            mode='min',
            factor=0.7,
            patience=20,
            verbose=1,
            cooldown=1,
            min_delta=2*min_delta,
            min_lr=1E-6
        )
        callbacks.append(lr)
    # provide checkpoint_path to checkpoint model and to save models when finished training
    if checkpoint_path is not None:
        print(f"Checkpointing model in {checkpoint_path}/...")
        if force_save:
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            checkpoint_path = f"{checkpoint_path}/"
        else:
            if not os.path.exists(checkpoint_path):
                print(f"Checkpoint path: {checkpoint_path} doesn't exist.")

        checkpoint = keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_weights_only=True,
            monitor='val_l',
            mode='min',
            save_best_only=True,
            save_freq='epoch'
        )

        # output loss at every epoch to training.log file for analysis
        csv_logger = keras.callbacks.CSVLogger(
            checkpoint_path + 'training.log'
        )

        # profile training for analysis of memory/speed bottlenecks
        # also useful for evaluating training/validation loss
        tensorboard = tf.keras.callbacks.TensorBoard(
            log_dir=checkpoint_path + 'tensorboard_logs/',
            update_freq='epoch',
            profile_batch='100,200'
        )

        callbacks.extend([checkpoint, csv_logger, tensorboard])

    # training and saving model
    try:
        history = model.fit(
            X,
            Y,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.2,
            callbacks=callbacks,
            shuffle=True,
            verbose=2
        )
        if checkpoint_path is not None:
            model.save_weights(checkpoint_path + 'model_weights.h5')
            print(f"Model weights saved to {checkpoint_path}model_weights.h5")
            model.save(checkpoint_path + "model_data")
            print(f"Model saved to {checkpoint_path}model_data")

    # still save model even if ctrl+c
    except KeyboardInterrupt:
        print("\n")
        print("Interrupting training...")
        if checkpoint_path is not None:
            model.save_weights(checkpoint_path + 'model_weights.h5')
            print(f"Model weights saved to {checkpoint_path}model_weights.h5")
            model.save(checkpoint_path + "model_data")
            print(f"Model saved to {checkpoint_path}model_data")
            weights = model.get_weights()
        else:
            weights = model.get_weights()
        return weights
    return history

def preprocess_data(X, Y, mu_r, RF, S, loop_scaled):
    mom_scaler = layers.Normalization(axis=2)
    mom_scaler.adapt(X[:, 2:], batch_size=2**17)

    map_scaler = layers.Normalization(axis=1)
    map_scaler.adapt(RF, batch_size=2**17)

    mu_scaler = layers.Normalization(axis=None)
    mu_scaler.adapt(np.log(mu_r), batch_size=2**17)

    sij_scaler = layers.Normalization(axis=1)
    sij_scaler.adapt(np.log(S), batch_size=2**17)

    y_scaler = layers.Normalization(axis=None)
    y_scaler.adapt(Y, batch_size=2**17)
    y_scaled = y_scaler(Y)

    y_scaler_loop = layers.Normalization(axis=None)
    y_scaler_loop.adapt(loop_scaled, batch_size=2**17)
    y_loop = y_scaler_loop(loop_scaled)

    return mom_scaler, map_scaler, mu_scaler, sij_scaler, y_scaler, y_scaled, y_scaler_loop, y_loop