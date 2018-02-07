import os
import numpy as np
import keras
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras import metrics, optimizers, initializers
from keras.callbacks import ModelCheckpoint, Callback, ReduceLROnPlateau, EarlyStopping
from shutil import copyfile


class MetricHistory(Callback):
    def __init__(self, path, interval):
        super(MetricHistory, self).__init__()
        self.path = path
        self.interval = interval

    def on_train_begin(self, logs={}):
        self.train_loss = []
        self.val_loss = []

    def on_epoch_end(self, epoch, logs={}):
        self.train_loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))

        if (epoch+1) % self.interval == 0:
            np.savez_compressed(self.path, 
                                train_loss=np.array(self.train_loss),
                                val_loss=np.array(self.val_loss))


def create_export_path(mdl_name):
    model_path = os.path.join('models', mdl_name)

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    copyfile('train_model.py', os.path.join(model_path, 'code.py'))

    return model_path


def load_dataset(path, input_indices, output_indices):
    data = np.load(path)
    inputs = data[:, input_indices]
    targets = data[:, output_indices]
    return inputs, targets


def normalise_inputs(inputs):
    input_scaler = StandardScaler().fit(inputs)
    inputs_norm = input_scaler.transform(inputs)
    return inputs_norm, input_scaler


def scale_targets(targets):
    target_scaler = MinMaxScaler(feature_range=(-1, 1),).fit(targets)
    targets_norm = target_scaler.transform(targets)
    return targets_norm, target_scaler


def partition_dataset(inputs, targets, val_size=0.1, random_seed=None):
    return train_test_split(inputs,
                            targets,
                            test_size=val_size,
                            random_state=random_seed)


def create_neural_net(input_dim, output_dim, n_hidden_layers, 
                      n_units_per_layer, random_seed=None):
    model = Sequential()

    model.add(Dense(n_units_per_layer, 
                    input_dim=input_dim,
                    kernel_initializer=initializers.glorot_uniform(random_seed),
                    bias_initializer='zeros'))
    model.add(Activation('relu'))

    for _ in range(n_hidden_layers-1):
        model.add(Dense(n_units_per_layer, 
                  kernel_initializer=initializers.glorot_uniform(random_seed),
                  bias_initializer='zeros'))
        model.add(Activation('relu'))

    model.add(Dense(output_dim, 
                kernel_initializer=initializers.glorot_uniform(random_seed),
                bias_initializer='zeros'))
    model.add(Activation('tanh'))

    # Learning rule
    optimizer = optimizers.Adam(lr=1e-3)
    model.compile(loss='mean_squared_error',
                  optimizer=optimizer)

    return model


def get_callbacks_export_model(path):
    save_model_path = os.path.join(path, "best_model_train.h5")

    checkpoint_best_train = ModelCheckpoint(save_model_path, 
                                 monitor='loss', 
                                 save_best_only=True,
                                 mode='min',
                                 period=1)

    save_model_path = os.path.join(model_path, "best_model_val.h5")
    checkpoint_best_val = ModelCheckpoint(save_model_path, 
                                 monitor='val_loss', 
                                 save_best_only=True,
                                 mode='min',
                                 period=1)

    return [checkpoint_best_train, checkpoint_best_val]


def get_callback_save_metrics(path):
    return MetricHistory(os.path.join(path, 'metrics'), 1)


def get_callback_reduce_lr():
    return ReduceLROnPlateau(monitor='loss',
                             factor=0.1,
                             patience=10,
                             mode='min',
                             epsilon=0.0001,
                             verbose=1)


def get_callback_early_stopping():
    return EarlyStopping(monitor='loss',
                         min_delta=0.0001,
                         patience=50,
                         verbose=1,
                         mode='min')


def get_callbacks(path):
    callbacks = get_callbacks_export_model(path)
    callbacks.append(get_callback_save_metrics(path))
    callbacks.append(get_callback_reduce_lr())
    callbacks.append(get_callback_early_stopping())
    return callbacks


def train(data, model, epochs=5, callbacks=None):
    (X_train, X_val, y_train, y_val) = data
    return model.fit(x=X_train,
                     y=y_train, 
                     epochs=epochs, 
                     batch_size=64,
                     validation_data=(X_val, y_val), 
                     callbacks=callbacks)


if __name__ == '__main__':
    random_seed = 170118

    ####################
    mdl_name = 'moc_all'

    data_path = 'datasets/moc_data_sph.npy'
    input_indices = [0,1,2,3,4,5,6]
    output_indices = [7,8,9] 
    n_hidden_layers = 1 # 4
    n_units_per_layer = 50 # 200
    ####################

    model_path = create_export_path(mdl_name)
    inputs, targets = load_dataset(data_path, input_indices, output_indices)
    inputs_norm, _ = normalise_inputs(inputs)
    targets_norm, _ = scale_targets(targets)

    X_train, X_val, y_train, y_val = partition_dataset(inputs_norm, 
                                                       targets_norm,
                                                       val_size=0.1,
                                                       random_seed=random_seed)

    model = create_neural_net(len(input_indices), 
                              len(output_indices),
                              n_hidden_layers,
                              n_units_per_layer,
                              random_seed)

    callbacks = get_callbacks(model_path)

    train((X_train, X_val, y_train, y_val),
          model,
          epochs=5000,
          callbacks=callbacks)
