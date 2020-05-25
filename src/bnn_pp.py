#!usr/bin/env python3

import keras
import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import argparse
from keras import backend as K
from bnn_dg import BubbleDirReader, vr_nn_seqgen, scalers


def parseArgs():
    """
    Argument parser for easy execution of script from terminal.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("FNM", help="Name of .hdf5 model for postprocessing")
    parser.add_argument("-c", "--csvdata", help="Csv of error and times",
                        action="store_true")
    return parser.parse_args()


def map_error(y_true, y_pred, err_typ="err"):
    """
    Function for getting mean percentage error and
    standard deviation of percentage error
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    if err_typ == "err":
        output = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    elif err_typ == "std":
        output = np.std(np.abs((y_true - y_pred) / y_true)) * 100
    return output


def prep_data(dir_name, gs, depth, scaler,
              segBool=True):
    """
    Function to load data that will be tested by saved NN.
    """
    def load(dat):
        # Load each dataset and normalize
        dat.genchunks(depth)
        xt, yt, _, _ = dat.load_data(tr_ratio=1)
        xt = scaler.void_scale(xt)
        xt = xt.reshape(xt.shape + (1,))  # Only 1 "channel"
        yt = scaler.vel_scale(yt)
        return [xt, yt]

    def load_seg(dat):
        # Load each (vg, vf) experiment separately
        dat.genchunks(depth)
        xt, yt, _, _ = dat.load_data(segBool=True, tr_ratio=1)
        xt = scaler.void_scale(xt)
        for i in range(0, xt.shape[0]):
            yt[i] = scaler.vel_scale(yt[i])
        xt = xt.reshape(xt.shape + (1,))  # Only 1 "channel"
        return [xt, yt]

    dat_lvf = BubbleDirReader(dir_name, gs, expsel='ex_low_vf')
    dat_mvf = BubbleDirReader(dir_name, gs, expsel='ex_med_vf')
    dat_hvf = BubbleDirReader(dir_name, gs, expsel='ex_high_vf')
    dat_hvg = BubbleDirReader(dir_name, gs, expsel='ex_high_vg')
    ls_out = []
    if not segBool:
        ls_out.append(('low_vf', load(dat_lvf)))
        ls_out.append(('med_vf', load(dat_mvf)))
        ls_out.append(('high_vf', load(dat_hvf)))
        ls_out.append(('high_vg', load(dat_hvg)))
    else:
        ls_out.append(('low_vf', load_seg(dat_lvf)))
        ls_out.append(('med_vf', load_seg(dat_mvf)))
        ls_out.append(('high_vf', load_seg(dat_hvf)))
        ls_out.append(('high_vg', load_seg(dat_hvg)))
    return ls_out


def model_mape(model, dir_name, gs, depth,
               lookback, batch_size, chBool,
               vch=None, segBool=True, verboseLvl=1):
    """
    Function to evaluate performance on all exp datasets
    Inputs:
        model: keras model to be evaluated
        dir_name: where exp data exists
        gs: grid size
        depth: chunk size for samples
        lookback: time-span trained for
        batch_size: batch-size
        chBool: True if last channel is (1,)
        vch: None if (vg, vf) else, 'vg' or 'vf' only
        segBool: if (vg, vf) permutaiton evaluated separately
        verboseLvl: Change level of verbosity (0:None, 1:Medium, >1:All)
    """
    lnverr_ls = []
    verr_ls = []
    vstd_ls = []
    tps_ls = []
    if vch is None:
        scaler = scalers()
    elif vch == 'vg':
        scaler = scalers(vch=vch)
    elif vch == 'vf':
        scaler = scalers(vch=vch)
    #
    ls_dat = prep_data(dir_name, gs, depth, scaler, segBool)
    for i in ls_dat:
        print('\033[1;33;40mEvaluating {}\033[1;37;40m'.format(i[0]))
        for k in range(i[1][0].shape[0]):
            # Information about loaded chunk
            print('\033[1;35;40mEvaluating {}\033[1;37;40m'
                  .format(i[1][1][k][0]))
            print('Mean gen void: {0:1.5e}'.format(i[1][0][k].mean()))

            # Loss calculation using keras built-in evaluation
            eval_gen = vr_nn_seqgen(i[1][0][k], i[1][1][k], lookback,
                                    vch=vch, batch_size=batch_size,
                                    chBool=chBool, randBool=False)
            spe = len(eval_gen)
            # time per void fraction matrix
            start_time = time.time()
            loss_builtin = model.evaluate_generator(eval_gen, steps=spe)
            end_time = time.time()
            tps_ls.append((end_time-start_time)/(spe*batch_size))
            #
            print('Built-in Keras Loss was {}'.format(loss_builtin))

            # Loss calculation manually
            err_gen = vr_nn_seqgen(i[1][0][k], i[1][1][k], lookback,
                                   vch=vch, batch_size=batch_size,
                                   chBool=chBool, randBool=False)
            y_pred = model.predict_generator(err_gen, steps=spe)
            # fill ground truth matrix
            y_true = []
            for idx in range(spe):
                _, y_temp = eval_gen.__getitem__(idx)
                if vch is not None:
                    y_temp = y_temp.reshape(y_temp.shape + (1,))
                y_true.append(y_temp)
            y_true = np.vstack(y_true)
            # get errors for this (vg,vf) point
            if vch is None:  # If both (vg, vf) were targets
                err_vg = map_error(y_true[:, 0], y_pred[:, 0])
                err_vf = map_error(y_true[:, 1], y_pred[:, 1])
                if verboseLvl > 1:
                    print('For ln(vg) {} map was {}'
                          .format(i[1][1][k][0], err_vg))
                    print('For ln(vf) {} map was {}'
                          .format(i[1][1][k][0], err_vf))
                lnverr_ls.append([i[1][1][k][0][0], i[1][1][k][0][1],
                                  err_vg, err_vf])
                # Error for descaled targets
                y_true = scaler.vel_descale(y_true)
                y_pred = scaler.vel_descale(y_pred)
                err_vg = map_error(y_true[:, 0], y_pred[:, 0])
                std_vg = map_error(y_true[:, 0], y_pred[:, 0], err_typ='std')
                err_vf = map_error(y_true[:, 1], y_pred[:, 1])
                std_vf = map_error(y_true[:, 1], y_pred[:, 1], err_typ='std')
                if verboseLvl > 0:
                    print('For vg {} map was {}'.format(i[1][1][k][0], err_vg))
                    print('For vf {} map was {}'.format(i[1][1][k][0], err_vf))
                verr_ls.append([i[1][1][k][0][0], i[1][1][k][0][1],
                                err_vg, err_vf])
                vstd_ls.append([i[1][1][k][0][0], i[1][1][k][0][1],
                                std_vg, std_vf])
            elif vch == 'vg' or vch == 'vf':
                err_v = map_error(y_true, y_pred)
                if verboseLvl > 1:
                    print('For ln({}) {} map was {3:1.2f}'
                          .format(vch, i[1][1][k][0], err_v))
                lnverr_ls.append([i[1][1][k][0][0], i[1][1][k][0][1], err_v])
                # Error for descaled targets
                y_true = scaler.vel_descale(y_true)
                y_pred = scaler.vel_descale(y_pred)
                err_v = map_error(y_true, y_pred)
                std_v = map_error(y_true, y_pred, err_typ='std')
                if verboseLvl > 0:
                    print('For {0} {1} true mu_v was {2:1.5f}'
                          .format(vch, i[1][1][k][0], y_true.mean()))
                    print('For {0} {1} pred mu_v was {2}'
                          .format(vch, i[1][1][k][0], y_pred.mean()))
                    print('For {0} {1} map was {2:1.2f}'
                          .format(vch, i[1][1][k][0], err_v))
                verr_ls.append([i[1][1][k][0][0], i[1][1][k][0][1], err_v])
                vstd_ls.append([i[1][1][k][0][0], i[1][1][k][0][1], std_v])
            # Deleting references to previous generators
            eval_gen = None
            err_gen = None

    lnverr = np.array(lnverr_ls)
    verr = np.array(verr_ls)
    vstd = np.array(vstd_ls)
    tps = np.array(tps_ls)
    return lnverr, verr, vstd, tps


if __name__ == "__main__":
    args = parseArgs()

    # Model Settings
    dir_name = 'Raw_Data/DN50/03_06_03_dual_50_Reihe_A_LD04'
    gs = 16
    depth = 50*gs
    vch = None

    # Generator settings
    chBool = True
    lookback = 256*2
    batch_size = 2**3

    # Configure tf memory to grow instead of allocate all
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)

    # Load NN model
    fileName = args.FNM + '.hdf5'
    model = keras.models.load_model(fileName)
    model.summary()
    print('\033[1;33;40mLOADED MODEL {}\033[1;37;40m'.format(fileName))

    scaler = scalers()

    # Get errorrs
    lnverr, verr, vstd, tps = model_mape(model, dir_name, gs, depth,
                                         lookback, batch_size, chBool,
                                         vch=vch)
    # Print csvs
    if args.csvdata:
        # csv of error
        fn = "{}_err.csv".format(args.FNM)
        verr[:, 0:2] = scaler.vel_descale(verr[:, 0:2])
        np.savetxt(fn, verr, delimiter=',', fmt='%.8e')
        print("Printed {} verr!".format(args.FNM))
        # csv of std
        fn = "{}_std.csv".format(args.FNM)
        vstd[:, 0:2] = scaler.vel_descale(vstd[:, 0:2])
        np.savetxt(fn, vstd, delimiter=',', fmt='%.8e')
        print("Printed {} vstd!".format(args.FNM))
        # csv of time taken
        fn = "{}_time.csv".format(args.FNM)
        tmat = np.hstack((verr[:, 0:2], tps.reshape(-1, 1)))
        np.savetxt(fn, tmat, delimiter=',', fmt='%.8e')
        print("Printed {} times!".format(args.FNM))
    # view charts of graph
    else:
        input("Press Enter to plot error...")
        if vch is None:
            fig, ax = plt.subplots(2, 1)
            cm = plt.cm.get_cmap('RdYlBu')
            im = ax[0].scatter(verr[:, 0], verr[:, 1], c=verr[:, 2], s=75)
            ax[0].set_title('vg Error')
            fig.colorbar(im, ax=ax[0])
            im = ax[1].scatter(verr[:, 0], verr[:, 1], c=verr[:, 3], s=75)
            ax[1].set_title('vf Error')
            fig.colorbar(im, ax=ax[1])
            plt.show()
        elif vch == 'vg' or vch == 'vf':
            fig, ax = plt.subplots(1, 1)
            cm = plt.cm.get_cmap('RdYlBu')
            im = ax.scatter(verr[:, 0], verr[:, 1], c=verr[:, 2], s=75)
            ax.set_title('{} Error'.format(vch))
            fig.colorbar(im, ax=ax)
            plt.show()
        input("\033[1;31;40mPress Enter to EXIT.\033[1;37;40m")
