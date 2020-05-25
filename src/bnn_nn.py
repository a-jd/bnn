#!usr/bin/env python3

import keras
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import backend as K
from bnn_dg import BubbleDirReader, vr_nn_seqgen
from bnn_dg import blockPrint, enablePrint, scalers

# Aliases foor keras layers
Input = keras.layers.Input
TDL = keras.layers.TimeDistributed
Conv3D = keras.layers.Conv3D
Dense = keras.layers.Dense
Flatten = keras.layers.Flatten
LSTM = keras.layers.LSTM

# Name for trained network (used for saving/loading hdf5)
_PRJNM = 'E_v1'


# Define custom callback to include folder name
class CustCheckpoint(keras.callbacks.ModelCheckpoint):
    def __init__(self, folder_name=None):
        fn = folder_name+'_saved-{epoch:02d}.hdf5'
        super().__init__(fn,
                         verbose=1,
                         period=10)


callbacks = [
    CustCheckpoint(_PRJNM)
]

# Configure tf memory to grow instead of allocate all
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)


def reload_data(dir_name, gs, expsel, shuffleBool=True):
    """
    Function to (re)load input data during training process
    """
    # Read input data
    bub_data = BubbleDirReader(dir_name, gs, expsel=expsel)
    bub_data.genchunks(depth)
    x_train, y_train, x_test, y_test = \
        bub_data.load_data(shuffleBool=shuffleBool)
    # Prepare training & test data
    scaler = scalers()
    x_train = scaler.void_scale(x_train)
    x_test = scaler.void_scale(x_test)
    x_train = x_train.reshape(x_train.shape + (1,))  # Only 1 "channel"
    x_test = x_test.reshape(x_test.shape + (1,))
    y_train = scaler.vel_scale(y_train)
    y_test = scaler.vel_scale(y_test)
    return x_train, y_train, x_test, y_test


##################################################
# Network Architecture Definitions


def model_A(sample_shape, nflt, sflt):
    """
    Only use dense layers
    """
    # Input
    input_void = Input(sample_shape[1:])

    # Dense
    x = Flatten()(input_void)
    x = Dense(8*nflt, activation='relu')(x)
    x = Dense(4*nflt, activation='relu')(x)
    output = Dense(2)(x)

    model = keras.Model(input_void, output)
    model.compile(optimizer='adam', loss='mape',
                  metrics=['mse', 'msle'])
    return model


def model_B(sample_shape, nflt, sflt):
    """
    Use TDL dense layers
    """
    # Input
    input_void = Input(sample_shape[1:])

    # TDL Dense
    x = TDL(Flatten())(input_void)
    x = TDL(Dense(8*nflt, activation='relu'))(x)
    x = TDL(Dense(8*nflt, activation='relu'))(x)
    x = Flatten()(x)
    output = Dense(2)(x)

    model = keras.Model(input_void, output)
    model.compile(optimizer='adam', loss='mape',
                  metrics=['mse', 'msle'])
    return model


def model_C(sample_shape, nflt, sflt):
    """
    Use TDL dense and LSTM layers
    """
    # Input
    input_void = Input(sample_shape[1:])

    # Main tower
    x = TDL(Flatten())(input_void)
    x = TDL(Dense(8*nflt, activation='relu'))(x)
    output = LSTM(2, activation='relu')(x)

    model = keras.Model(input_void, output)
    model.compile(optimizer='adam', loss='mape',
                  metrics=['mse', 'msle'])
    return model


def model_D(sample_shape, nflt, sflt):
    """
    Conv head with Dense tail
    """
    # Model Settings
    zm = 3  # z-axis multiplier
    mflt = (zm*sflt, sflt, sflt)

    # Main tower
    input_void = Input(sample_shape[1:])
    x = Conv3D(nflt, mflt, padding='same', activation='relu',
               strides=(3*2, 2, 2))(input_void)
    x = Conv3D(nflt, mflt, padding='same', activation='relu',
               strides=(3*2, 2, 2))(x)
    x = Flatten()(x)
    x = Dense(nflt, activation='relu')(x)
    output = Dense(2)(x)

    model = keras.Model(input_void, output)
    model.compile(optimizer='adam', loss='mape',
                  metrics=['mse', 'msle'])
    return model


def model_E(sample_shape, nflt, sflt):
    """
    Taking only one Conv train and split into two LSTMs
    """
    # Model Settings
    zm = 3  # z-axis multiplier
    mflt = (zm*sflt, sflt, sflt)

    # Input
    input_void = Input(sample_shape[1:])

    # Main tower
    x = Conv3D(nflt, mflt, padding='same', activation='relu')(input_void)
    x = Conv3D(nflt, mflt, padding='same', activation='relu',
               strides=(zm*2, 2, 2))(x)
    x = TDL(Flatten())(x)
    x = TDL(Dense(8*nflt, activation='relu'))(x)

    # LSTM split
    xgf = LSTM(1)(x)
    xff = LSTM(1)(x)

    # Output
    output = keras.layers.Concatenate()([xgf, xff])

    model = keras.Model(input_void, output)
    model.compile(optimizer='adam', loss='mape',
                  metrics=['mse', 'msle'])
    return model


def model_F(sample_shape, nflt, sflt):
    """
    Two fully separate Conv trains
    """
    # Model Settings
    zm = 3  # z-axis multiplier
    mflt = (zm*sflt, sflt, sflt)

    # Input
    input_void = Input(sample_shape[1:])

    # vg tower
    xg = Conv3D(nflt, mflt, padding='same', activation='relu')(input_void)
    xg = Conv3D(nflt, mflt, padding='same', activation='relu',
                strides=(zm*2, 2, 2))(xg)
    xg = TDL(Flatten())(xg)
    xg = TDL(Dense(8*nflt, activation='relu'))(xg)
    xg = LSTM(1)(xg)

    # vf tower
    xf = Conv3D(nflt, mflt, padding='same', activation='relu')(input_void)
    xf = Conv3D(nflt, mflt, padding='same', activation='relu',
                strides=(zm*2, 2, 2))(xf)
    xf = TDL(Flatten())(xf)
    xf = TDL(Dense(8*nflt, activation='relu'))(xf)
    xf = LSTM(1)(xf)

    # Output
    output = keras.layers.Concatenate()([xg, xf])

    model = keras.Model(input_void, output)
    model.compile(optimizer='adam', loss='mape',
                  metrics=['mse', 'msle'])
    return model


##################################################
# Script for training network


# Data Settings
dir_name = 'Raw_Data/DN50/03_06_03_dual_50_Reihe_A_LD04'
gs = 16  # grid size
expsel = 'all'  # custom exp selection
lookback = 512  # length of each sample
depth = 50*gs  # tootal depth of each chunk
x_train, y_train, x_test, y_test = reload_data(dir_name, gs, expsel)

# Network settings
batch_size = 2**7
sflt = 3  # Shape of filters
nflt = 16  # Number of filters
epochs = 100

# Setup
model_ch = model_E
vch = None  # if training for a single vel, else input as None

# Generators
train_gen = vr_nn_seqgen(x_train, y_train, lookback,
                         vch=vch, batch_size=batch_size)
val_gen = vr_nn_seqgen(x_test, y_test, lookback,
                       vch=vch, batch_size=batch_size)

(x_tc, y_tc) = train_gen.__getitem__(0)
spe = len(train_gen)
vpe = len(val_gen)

# Initialize/load model
history_ls = []
initial_epoch = 0  # if restarting from saved model
if initial_epoch == 0:
    model = model_ch(x_tc.shape, nflt, sflt)
else:
    fileName = 'saved-{epoch:02d}.hdf5'.format(epoch=initial_epoch)
    print('\033[1;33;40mLOADED MODEL {}\033[1;37;40m'.format(fileName))
    model = keras.models.load_model(fileName)

# Plot keras model graph
model.summary()

# Start Training
if initial_epoch < epochs:
    for _ in range(epochs):
        history = model.fit_generator(train_gen, steps_per_epoch=spe,
                                      epochs=initial_epoch+1,
                                      validation_data=val_gen,
                                      validation_steps=vpe,
                                      initial_epoch=initial_epoch,
                                      use_multiprocessing=True,
                                      callbacks=callbacks,
                                      workers=8)
        history_ls.append(history)
        initial_epoch += 1
        # Remix training & val data
        if initial_epoch % 20 == 0:
            blockPrint()
            x_train, y_train, x_test, y_test = \
                reload_data(dir_name, gs, expsel)
            enablePrint()
            print('\033[1;33;40mRELOADED DATA.\033[1;37;40m')
            train_gen = vr_nn_seqgen(x_train, y_train, lookback,
                                     vch=vch, batch_size=batch_size)
            val_gen = vr_nn_seqgen(x_test, y_test, lookback,
                                   vch=vch, batch_size=batch_size)
    # Loss information
    input("Press Enter to plot history...")
    loss = []
    val_loss = []
    for hist in history_ls:
        loss.append(hist.history['loss'])
        val_loss.append(hist.history['val_loss'])
    last_epoch = history_ls[-1].epoch[0]+1
    first_epoch = last_epoch - len(history_ls)
    epoch_range = range(first_epoch+1, last_epoch+1)
    plt.figure()
    plt.plot(epoch_range, loss, 'r', label='Training loss')
    plt.plot(epoch_range, val_loss, '--b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    input("Press Enter to plot figures...")
    plt.show()
else:
    print('initial_epoch set as > epochs (total epochs)')

input("\033[1;31;40mPress Enter to EXIT.\033[1;37;40m")
