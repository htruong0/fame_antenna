import warnings
from tensorflow import keras
import numpy as np

class RatioEarlyStopping(keras.callbacks.Callback):
    def __init__(self, ratio=0.0,
                 patience=0, verbose=0, restore_best_weights=True):
        super(RatioEarlyStopping, self).__init__()

        self.ratio = ratio
        self.patience = patience
        self.verbose = verbose
        self.wait = 0
        self.stopped_epoch = 0
        self.monitor_op = np.greater
        self.best_weights = None
        self.restore_best_weights = restore_best_weights

    def on_train_begin(self, logs=None):
        self.wait = 0  # Allow instances to be re-used

    def on_epoch_end(self, epoch, logs=None):
        current_val = logs.get('val_l')
        current_train = logs.get('l')
        if current_val is None:
            warnings.warn('Early stopping requires %s available!' %
                          (self.monitor), RuntimeWarning)

        # If ratio current_loss / current_val_loss > self.ratio
        if self.monitor_op(np.divide(current_train,current_val),self.ratio):
            self.wait = 0
            self.best_weights = self.model.get_weights()
            self.best_epoch = epoch
        else:
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                if self.restore_best_weights and self.best_weights is not None:
                    if self.verbose > 0:
                        print(
                            "Restoring model weights from "
                            "the end of the best epoch: "
                            f"{self.best_epoch + 1}."
                        )
                    self.model.set_weights(self.best_weights)
            self.wait += 1
            

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print('Epoch %05d: ratio early stopping' % (self.stopped_epoch))