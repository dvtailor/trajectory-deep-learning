import os
import numpy as np
import keras
import pykep
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras import metrics, regularizers, optimizers, initializers
from keras.layers.normalization import BatchNormalization
import keras.backend as K
from keras.callbacks import ModelCheckpoint, Callback, ReduceLROnPlateau, EarlyStopping
from keras.models import load_model
from imblearn.over_sampling import RandomOverSampler 
from shutil import copyfile

rnd_seed = 170118

MDL_NAME = 'moc_l4_n200_vector_theta'

n_hidden_layers = 4
n_units_per_layer = 200
enable_batch_norm = False
act_func = 'relu'
kernel_initializer = initializers.glorot_uniform(rnd_seed)
shuffle = 'batch' # True
augment_dataset = None # 1 (Pos/Vel mag) 2 (Kep elems) # 3 (All)


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

model_path = 'models/' + MDL_NAME

if not os.path.exists(model_path):
    os.makedirs(model_path)

copyfile('train_vector_network.py', model_path + '/code.py')


# Prepare data
moc_data = np.load('moc_data.npy')

input_indices = [0,1,2,3,4,5,6]
input_dim = 7

output_indices = [9] # {phi=8, theta=9}
output_dim = 1 # 2

inputs = moc_data[:, input_indices]

# STATE TRANSFORMATION

# Compute additional variables
n_instances = len(inputs)
# Compute vector norms of position and velocity
r_mag = np.linalg.norm(inputs[:,:3], axis=1)
v_mag = np.linalg.norm(inputs[:,3:6], axis=1)
# Compute keplerian elems
inputs_kep = np.array([pykep.ic2par(inputs[i,:3], inputs[i,3:6], pykep.MU_SUN) for i in range(n_instances)])

# Augment state with 2 variables: position & velocity magnitude
if augment_dataset == 1:
    input_dim = 9
    inputs_aug = np.zeros((n_instances, input_dim))
    inputs_aug[:, :3] = inputs[:,:3]
    inputs_aug[:, 3] = r_mag
    inputs_aug[:, 4:7] = inputs[:,3:6]
    inputs_aug[:, 7] = v_mag
    inputs_aug[:, 8] = inputs[:,6]

    inputs = inputs_aug

# Transform dataset to keplerian elements
if augment_dataset == 2:
    input_dim = 7
    inputs_aug = np.zeros((n_instances, input_dim))
    inputs_aug[:, :6] = inputs_kep
    inputs_aug[:, 6] = inputs[:, 6]

    inputs = inputs_aug

# Augment state with (r,v) mags & kep elems
if augment_dataset == 3:
    input_dim = 15
    inputs_aug = np.zeros((n_instances, input_dim))
    inputs_aug[:, :3] = inputs[:,:3]    # position
    inputs_aug[:, 3] = r_mag            # pos mag
    inputs_aug[:, 4:7] = inputs[:,3:6]  # velocity
    inputs_aug[:, 7] = v_mag            # vel mag
    inputs_aug[:, 8:14] = inputs_kep    # keplerian
    inputs_aug[:, 14] = inputs[:, 6]    # mass

    inputs = inputs_aug

targets = moc_data[:, output_indices]

input_scaler = StandardScaler().fit(inputs)
inputs_norm = input_scaler.transform(inputs)

target_scaler = MinMaxScaler(feature_range=(-1, 1),).fit(targets)
targets_norm = target_scaler.transform(targets)

X_train, X_test, y_train, y_test = train_test_split(inputs_norm, targets_norm, test_size=0.1, random_state=rnd_seed)


# Setup model
model = Sequential()

model.add(Dense(n_units_per_layer, 
                input_dim=input_dim,
                kernel_initializer=kernel_initializer,
                bias_initializer='zeros'))
if enable_batch_norm:
    model.add(BatchNormalization())
model.add(Activation(act_func))

for i in range(n_hidden_layers-1):
    model.add(Dense(n_units_per_layer, 
                    kernel_initializer=kernel_initializer,
                    bias_initializer='zeros'))
    if enable_batch_norm:
        model.add(BatchNormalization())
    model.add(Activation(act_func))

model.add(Dense(output_dim, 
                kernel_initializer=kernel_initializer,
                bias_initializer='zeros'))
if enable_batch_norm:
    model.add(BatchNormalization())
model.add(Activation('tanh'))


# Learning rule
optimizer = optimizers.Adam(lr=1e-3)
# optimizer = optimizers.Adam(lr=1e-2)
# optimizer = optimizers.SGD(lr=1e-4, momentum=0.9, nesterov=True)

model.compile(loss='mean_squared_error',
              optimizer=optimizer)

# Callbacks
# Save model (with weights)
callbacks = []
save_model_path = model_path + "/best_model_train.h5"
checkpoint_best_train = ModelCheckpoint(save_model_path, 
                             monitor='loss', 
                             save_best_only=True,
                             mode='min',
                             period=1)
callbacks.append(checkpoint_best_train)

save_model_path = model_path + "/best_model_val.h5"
checkpoint_best_val = ModelCheckpoint(save_model_path, 
                             monitor='val_loss', 
                             save_best_only=True,
                             mode='min',
                             period=1)
callbacks.append(checkpoint_best_val)

# save_model_path = model_path + "/exported_models/model_{epoch:04d}.h5"
# checkpoint_interval = ModelCheckpoint(save_model_path, 
#                              period=25)
# callbacks.append(checkpoint_interval)

# Save train/val performance
save_interval = 25
metrics = MetricHistory(model_path + '/metrics', save_interval)
callbacks.append(metrics)

# Reduce learning rate
reduce_lr = ReduceLROnPlateau(monitor='loss',
                    factor=0.1,
                    patience=10, # CHANGE
                    mode='min',
                    epsilon=0.0001,
                    verbose=1)
callbacks.append(reduce_lr)

# Early stopping
early_stop = EarlyStopping(monitor='loss',
                    min_delta=0.0001,
                    patience=50,
                    verbose=1,
                    mode='min')
callbacks.append(early_stop)


# Train
history_callback = model.fit(X_train,
                             y_train, 
                             epochs=5000, 
                             batch_size=64,
                             validation_data=(X_test, y_test), 
                             shuffle=shuffle,
                             callbacks=callbacks)
