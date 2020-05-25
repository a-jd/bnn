#!/usr/bin/env python3

import math
import sys
import numpy as np
from os import listdir, devnull
from os.path import isfile, join, splitext, basename
from sklearn.preprocessing import MinMaxScaler
from keras.utils import Sequence
import re


class scalers():
    """
    Class to provide scaling of superficial velocity and void fraction
    WARNING: Values are hardcoded according to WMS DB
    Inputs:
        vch: If training training for only 'vg' or 'vf'
    """
    def __init__(self, vch=None):
        # Database constants
        min_vel_vg = 0.0235
        max_vel_vg = 4.975
        min_vel_vf = 0.0405
        max_vel_vf = 2.554
        self.scaler = MinMaxScaler((0.1, 0.9))  # Will scale to range(0.1, 0.9)
        if vch is None:
            vel_ranges = [[min_vel_vg, min_vel_vf], [max_vel_vg, max_vel_vf]]
            self.scaler.fit(np.log(vel_ranges))
        elif vch == 'vg':
            # Reshape required by scaler.fit for single feature
            vel_ranges = [min_vel_vg, max_vel_vg]
            self.scaler.fit(np.log(vel_ranges).reshape(-1, 1))
        elif vch == 'vf':
            vel_ranges = [min_vel_vf, max_vel_vf]
            self.scaler.fit(np.log(vel_ranges).reshape(-1, 1))

    def void_scale(self, in_arr):
        return in_arr/100.0

    def void_descale(self, in_arr):
        return in_arr*100.0

    def vel_scale(self, in_arr):
        return self.scaler.transform(np.log(in_arr))

    def vel_descale(self, in_arr):
        return np.exp(self.scaler.inverse_transform(in_arr))


class BubbleReader():
    """
    Class to read binary WMS void fraction files
    Methods:
        readVoid: method to read the binary WMS file
        set_depth_size: sets depth size for creating samples
        depth_sort: creates samples based on deptth size
        print_sorted_stats: prints stats for debugging
    """
    MAX_VOID = 100  # for removing any previous masks
    DEFAULT_DEPTH = 250  # default for depth_sort

    def __init__(self, fileName, gs, vg, vf,
                 removeMaskFlag=True):
        '''
        Inputs:
            fileName: filename to be read without .v or .b specified
            gs: WMS grid size (e.g. 16, 64)
        '''
        self._depth_size = self.DEFAULT_DEPTH
        self._gridSize = gs
        self.vg = vg
        self.vf = vf
        self.readVoid(fileName, self._gridSize, removeMaskFlag)  # Store void

    def readVoid(self, toRead, gs, removeMaskFlag):
        '''
        Read the void fraction file (verfied vs. ipynb)
        Inputs:
            toRead: full file path to be read *.v
            gs: grid size
            removeMaskFlag: if removing any masks
        '''
        # dummStore created in loop, because passed by reference.
        toReadVoid = toRead+'.v'
        voidMatr = []
        # Fill matrix
        # Definition of lambda function below avoids EOF processing
        with open(toReadVoid, "rb") as binaFile:
            self._void_fn = toReadVoid
            for framByts in iter(lambda: binaFile.read(gs*gs), b""):
                newFrame = np.frombuffer(framByts, dtype=np.dtype('u1'))
                dummStore = newFrame.reshape(gs, gs)
                voidMatr.append(dummStore)
        # Store frame size and filled matrix
        self._framSize = len(voidMatr)
        voidMatr = np.array(voidMatr)
        if removeMaskFlag:
            voidMatr[voidMatr > self.MAX_VOID] = 0
        self._voidMatr = voidMatr

    def set_depth_size(self, depth):
        self._depth_size = depth

    def depth_sort(self):
        '''
        Split full void fraction matrix into smaller chunks/samples
        '''
        total_depth = self._voidMatr.shape[0]
        nseg = int(total_depth/self._depth_size)
        void = self._voidMatr
        void_sorted = []
        # Chunking:
        for i in range(nseg):
            void_sorted.append(void[i*self._depth_size:(i+1)*self._depth_size])
        print('Added {} segments for {}'.format(nseg, self._void_fn))
        self.void_sorted = void_sorted
        return None

    def print_sorted_stats(self):
        '''
        Summarize statistics of void chunks
        '''
        try:
            smat = self.void_sorted
        except AttributeError:
            print("Not yet chunked!")
        # Percentage of slices that are completely empty
        zero_mu = np.mean([~np.any(i)
                           for voids in smat
                           for i in voids])
        # Average void fraction in non-empty slices
        void_mu = np.mean([np.mean(j)
                           for voids in smat
                           for i in voids
                           for j in i[np.any(i)]])
        # Average void fraction in all slices
        vall_mu = np.mean(smat)
        print('Pct of slices that are empty: {0:3.2f}%'.format(zero_mu*100))
        print('Void frac of non-empty slices: {0:3.2f}/100'.format(void_mu))
        print('Void frac of all slices: {0:3.2f}/100'.format(vall_mu))


class BubbleDirReader():
    """
    Class to read entire WMS database and prepare data for training/testing
    Methods:
        genchunks: split full void fraction matrices into samples
        load_data: loader for training/testing
        printmats: output void fraction matrices for MATLAB postprocessing
    """
    # Dictionaries that will initiate input data for superficial velocities
    # Exp table available in https://doi.org/10.1016/j.ces.2017.01.001
    EXP_DICT = {
        # vf = 0.102 m/s
        'low_vf':  [58, 69, 80, 91, 102, 113, 124, 135, 146, 157,
                    168, 179, 190],
        # vf = 0.405 m/s
        'med_vf':  [61, 72, 83, 94, 105, 116, 127, 138, 149, 160,
                    171, 182, 193],
        # vf = 1.017 m/s
        'high_vf': [63, 74, 85, 96, 107, 118, 129, 140, 151, 162,
                    173, 184, 195],
        # vg = 0.218 m/s
        'high_vg': [111, 112, 113, 114, 115, 116, 117, 118, 119,
                    120]
    }
    #
    VEL_DICT = {
        # low_vf
        'low_vf_vg': np.logspace(math.log(0.0235), math.log(4.975),
                                 len(EXP_DICT['med_vf']), base=math.exp(1)),
        'low_vf_vf': np.ones(len(EXP_DICT['med_vf']))*0.102,
        # med_vf
        'med_vf_vg': np.logspace(math.log(0.0235), math.log(4.975),
                                 len(EXP_DICT['med_vf']), base=math.exp(1)),
        'med_vf_vf': np.ones(len(EXP_DICT['med_vf']))*0.405,
        # high_vf
        'high_vf_vg': np.logspace(math.log(0.0235), math.log(4.975),
                                  len(EXP_DICT['high_vf']), base=math.exp(1)),
        'high_vf_vf': np.ones(len(EXP_DICT['high_vf']))*1.017,
        # high_vg
        'high_vg_vg': np.ones(len(EXP_DICT['high_vg']))*0.219,
        'high_vg_vf': np.logspace(math.log(0.0405), math.log(2.554),
                                  len(EXP_DICT['high_vg']), base=math.exp(1))
    }
    COMB_DICT = {
        'low_vf': dict(zip(EXP_DICT['low_vf'],
                           np.hstack((VEL_DICT['low_vf_vg'][np.newaxis].T,
                                     VEL_DICT['low_vf_vf'][np.newaxis].T)))),
        'med_vf': dict(zip(EXP_DICT['med_vf'],
                           np.hstack((VEL_DICT['med_vf_vg'][np.newaxis].T,
                                     VEL_DICT['med_vf_vf'][np.newaxis].T)))),
        'high_vf': dict(zip(EXP_DICT['high_vf'],
                            np.hstack((VEL_DICT['high_vf_vg'][np.newaxis].T,
                                      VEL_DICT['high_vf_vf'][np.newaxis].T)))),
        'high_vg': dict(zip(EXP_DICT['high_vg'],
                            np.hstack((VEL_DICT['high_vg_vg'][np.newaxis].T,
                                      VEL_DICT['high_vg_vf'][np.newaxis].T))))
    }

    def __init__(self, dirName, gs, expsel=None):
        self.dirName = dirName
        self.gs = gs
        self._getNames(self.dirName)
        self._exp_selector(expsel)
        self._readFiles()

    def _getNames(self, dirName):
        '''
        Fetches all .v files in dirName
        '''
        voidfn = [f for f in listdir(dirName) if isfile(join(dirName, f)) and
                  f.endswith(".v")]
        voidfn.sort()
        self.voidfn = voidfn

    def _exp_selector(self, exp_type):
        '''
        Allows easy selection of all experimental data available
        '''
        # chdict is the chosen dictionary:
        # the keys will be integer of experiment number
        # the value will be another dict containing fn, vg, vf
        def __custom_exp(ech):
            '''
            Choose user-defined experiment groups
            '''
            cdict = {
                # All exp for training
                'all': [58, 69, 80, 91, 102, 113, 124, 135, 146, 157, 168,
                        179, 190, 61, 72, 83, 94, 105, 116, 127, 138, 149,
                        160, 171, 182, 193, 63, 74, 85, 96, 107, 118, 129,
                        140, 151, 162, 173, 184, 195, 111, 112, 113, 114,
                        115, 116, 117, 118, 119, 120],
                # Sets for post-processing data
                'ex_low_vf': [58, 69, 80, 91, 102, 113, 124, 135, 146, 157,
                              168, 179, 190],
                'ex_med_vf': [61, 72, 83, 94, 105, 116, 127, 138, 149, 160,
                              171, 182, 193],
                'ex_high_vf': [63, 74, 85, 96, 107, 118, 129, 140, 151, 162,
                               173, 184, 195],
                'ex_high_vg': [111, 112, 113, 114, 115, 116, 117, 118, 119,
                               120]
            }
            return cdict[ech]

        if exp_type in self.COMB_DICT.keys():  # Default groups
            templs = [self.COMB_DICT[exp_type]]
            chdict = dict((key, d[key]) for d in templs for key in d)
            exp_choice = chdict.keys()
        else:  # All or custom groups
            templs = [self.COMB_DICT[k] for k in self.COMB_DICT]
            chdict = dict((key, d[key]) for d in templs for key in d)
            if exp_type == 'all':
                exp_choice = chdict.keys()
            else:
                exp_choice = __custom_exp(exp_type)

        # Removing file names that are not selected or listed in DICT
        tempfn = self.voidfn[:]
        for i, expfn in enumerate(tempfn):
            expid = int(re.search('\d\d\d', basename(expfn)).group())
            if expid in exp_choice:
                chdict[expid] = dict(fn=expfn,
                                     vg=chdict[expid][0],
                                     vf=chdict[expid][1])
            else:
                self.voidfn.remove(expfn)
                if expid in chdict:
                    del chdict[expid]
                print('Removed', expfn)
        self.chdict = chdict

    def _readFiles(self):
        '''
        Create a list of BubbleReader objects
        '''
        voidls = []  # List of BubbleReader objects
        for f in self.chdict.values():
            fileNm = splitext(f['fn'])[0]
            voidls.append(BubbleReader(join(self.dirName, fileNm), self.gs,
                                       f['vg'], f['vf']))
        self.voids = voidls

    def genchunks(self, depth=250):
        for b in self.voids:
            b.set_depth_size(depth)
            b.depth_sort()

    def load_data(self, tr_ratio=0.80,
                  singleBool=False, shuffleBool=True,
                  segBool=False):
        '''
        Take generated chunks and creates x, y train & validation sets
            :singleBool: if only one first sample needed (for debugging)
            :segBool: if (vg, vf) evaluated separately
        '''
        # x is void matrix, y is (vg, vf)
        xtrain_list = []
        ytrain_list = []
        xvalid_list = []
        yvalid_list = []
        for b in self.voids:
            ytup = (b.vg, b.vf)
            if hasattr(b, 'void_sorted'):  # void has been chunked
                addb = np.array(b.void_sorted)
                np.random.shuffle(addb)  # So that train/valid have no sequence
                if addb.size != 0:
                    train_lim = int(addb.shape[0]*tr_ratio)
                    valid_lim = addb.shape[0]-train_lim
                    xtrain_list.append(addb[0:train_lim])
                    ytrain_list.append(np.tile(ytup, (train_lim, 1)))
                    xvalid_list.append(addb[train_lim:])
                    yvalid_list.append(np.tile(ytup, (valid_lim, 1)))
                else:
                    print('WARNING: test {} ignored'.format(b._void_fn))
                    xtrain_list.append(None)
                    ytrain_list.append(None)
                    xvalid_list.append(None)
                    yvalid_list.append(None)
            else:  # Taking full void matrix
                # Need to add newaxis to match chunk dims
                xtrain_list.append(b._voidMatr[np.newaxis])
                ytrain_list.append(np.asarray(ytup)[np.newaxis])
                xvalid_list.append(b._voidMatr[np.newaxis])
                yvalid_list.append(np.asarray(ytup)[np.newaxis])
        if segBool:
            xtrain_arr = np.array(xtrain_list)
            ytrain_arr = np.array(ytrain_list)
            xvalid_arr = np.array(xvalid_list)
            yvalid_arr = np.array(yvalid_list)
        else:
            xtrain_arr = np.concatenate(xtrain_list, axis=0)
            ytrain_arr = np.concatenate(ytrain_list, axis=0)
            xvalid_arr = np.concatenate(xvalid_list, axis=0)
            yvalid_arr = np.concatenate(yvalid_list, axis=0)
            if shuffleBool:
                xtrain_arr, ytrain_arr = unison_shuffle(xtrain_arr, ytrain_arr)
                xvalid_arr, yvalid_arr = unison_shuffle(xvalid_arr, yvalid_arr)
            if singleBool:
                sidx = 0
                xtrain_arr = xtrain_arr[sidx][np.newaxis]
                ytrain_arr = ytrain_arr[sidx][np.newaxis]
                xvalid_arr = xvalid_arr[sidx][np.newaxis]
                yvalid_arr = yvalid_arr[sidx][np.newaxis]
        return (xtrain_arr, ytrain_arr, xvalid_arr, yvalid_arr)

    def printmats(self, expch, segmentBool=False,
                  depth=0, spacing=10):
        import scipy.io as scio
        voidmat = self.voids[expch]._voidMatr
        vg = self.voids[expch].vg
        vf = self.voids[expch].vf
        if not segmentBool:
            fn = 'Expch-{}_vg-{}_vf-{}.mat'.format(expch, vg, vf)
            scio.savemat(fn, {'vm': voidmat})
        else:
            for i in range(0, voidmat.shape[0]-depth, spacing):
                fn = 'i-{}_Expch-{}_vg-{}_vf-{}.mat'.format(i, expch, vg, vf)
                fn = 'output_folder/' + fn
                scio.savemat(fn, {'vm': voidmat[i:i+depth]})


class vr_nn_seqgen(Sequence):
    """
    Generator for velocity recognition networks
    Inputs:
        voids: void mat shape (nvoids, frames, gs, gs)
        vels: (vg, vf) shape (nvoids, 2)
        lookback: # of frames yielded (will scan over void frames)
        vch: None if both (vg, vf)
        batch_size: # of samples per yield
        randBool: shuffle the yield output
        chBool: if last shape (1,) to conform to keras input shape
    """
    def __init__(self, voids, vels, lookback, vch=None,
                 batch_size=1, randBool=True, chBool=True):
        self.x = voids
        self.y = vels
        self.lookback = lookback
        self.vch = vch
        self.batch_size = batch_size
        self.randBool = randBool
        self.chBool = chBool
        if not self.chBool:
            self.x = self.x.squeeze(axis=(4,))
        self._preprocess()

    def _preprocess(self):
        # Preprocessing:
        nvoids = self.x.shape[0]
        nframes = self.x.shape[1]
        svInputs = []
        svTargets = []
        # Generate all yield possibilities:
        i = self.lookback  # frame counter
        j = 0  # void counter
        while j < nvoids:
            indices = slice(i - self.lookback, i)
            inputs = [j, indices]
            targets = j
            # Reset frame ctr, increment ctrs
            i += 1
            if i > nframes:
                i = self.lookback
                j += 1
            # Sequential API requirement for yield
            svInputs.append(inputs)
            svTargets.append(targets)
        self.length = len(svInputs)
        self.xsvInp, self.ysvInp = (svInputs, svTargets)
        self.on_epoch_end()

    def on_epoch_end(self):
        if self.randBool:
            self.xInp, self.yInp = unison_shuffle(self.xsvInp, self.ysvInp)
        else:
            self.xInp, self.yInp = unison_shuffle(self.xsvInp, self.ysvInp,
                                                  seededBool=True)

    def __len__(self):
        return int(np.floor(self.length/self.batch_size))

    def __getitem__(self, idx):
        # Yield all inputs, targets
        k = idx*self.batch_size  # yield ctr
        idxs = slice(k, min(k+self.batch_size, self.length))
        # Fetch voids & vels
        void_out = np.concatenate([self.x[tuple(i)][np.newaxis]
                                   for i in self.xInp[idxs]], axis=0)
        vel_out = self.y[self.yInp[idxs]]
        if self.vch is not None:
            if self.vch == 'vg':
                vel_out = vel_out[:, 0]
            elif self.vch == 'vf':
                vel_out = vel_out[:, 1]
            else:
                raise NameError('Incorrect vch option.')
        return void_out, vel_out


def unison_shuffle(x_in, y_in, seededBool=False):
    """
    Shuffle x and y sets in unison
    Optional input seededBool changes if pseudorandom
    """
    assert len(x_in) == len(y_in)
    p = np.random.permutation(len(x_in))
    if seededBool:
        p = np.random.RandomState(seed=42).permutation(len(x_in))
    if type(x_in) == np.ndarray:
        x_out, y_out = x_in[p], y_in[p]
    elif type(x_in) == list:
        x_out = [x_in[i] for i in p]
        y_out = [y_in[i] for i in p]
    return x_out, y_out


def blockPrint():
    sys.stdout = open(devnull, 'w')


def enablePrint():
    sys.stdout = sys.__stdout__
