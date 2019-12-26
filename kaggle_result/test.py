#!/usr/bin/env python
# coding: utf-8

# In[232]:


import pandas as pd
import numpy as np
import sys
import os
sys.path.append('/home/aistudio/external-libraries')


# In[233]:


from collections.abc import Sequence
import random
import skimage


# In[234]:


# df = pd.read_csv('work/机器学习/train_val.csv')
result=pd.read_csv('sampleSubmission.csv')


# In[235]:


import collections
from itertools import repeat
import numpy as np
import scipy
import matplotlib.pyplot as plt
from skimage.measure import find_contours
def plot_voxel(arr, aux=None):
    if aux is not None:
        assert arr.shape == aux.shape
    length = arr.shape[0]
    _, axes = plt.subplots(length, 1, figsize=(4, 4 * length))
    for i, ax in enumerate(axes):
        ax.set_title("@%s" % i)
        ax.imshow(arr[i], cmap=plt.cm.gray)
        if aux is not None:
            ax.imshow(aux[i], alpha=0.3)
    plt.show()


def plot_voxel_save(path, arr, aux=None):
    if aux is not None:
        assert arr.shape == aux.shape
    length = arr.shape[0]
    for i in range(length):
        plt.clf()
        plt.title("@%s" % i)
        plt.imshow(arr[i], cmap=plt.cm.gray)
        if aux is not None:
            plt.imshow(aux[i], alpha=0.2)
        plt.savefig(path + "%s.png" % i)


def plot_voxel_enhance(arr, arr_mask=None, figsize=10, alpha=0.1):  # zyx
    '''borrow from yuxiang.'''
    plt.figure(figsize=(figsize, figsize))
    rows = cols = int(round(np.sqrt(arr.shape[0])))
    img_height = arr.shape[1]
    img_width = arr.shape[2]
    assert img_width == img_height
    res_img = np.zeros((rows * img_height, cols * img_width), dtype=np.uint8)
    if arr_mask is not None:
        res_mask_img = np.zeros(
            (rows * img_height, cols * img_width), dtype=np.uint8)
    for row in range(rows):
        for col in range(cols):
            if (row * cols + col) >= arr.shape[0]:
                continue
            target_y = row * img_height
            target_x = col * img_width
            res_img[target_y:target_y + img_height,
            target_x:target_x + img_width] = arr[row * cols + col]
            if arr_mask is not None:
                res_mask_img[target_y:target_y + img_height,
                target_x:target_x + img_width] = arr_mask[row * cols + col]
    plt.imshow(res_img, plt.cm.gray)
    if arr_mask is not None:
        plt.imshow(res_mask_img, alpha=alpha)
    plt.show()


def find_edges(mask, level=0.5):
    edges = find_contours(mask, level)[0]
    ys = edges[:, 0]
    xs = edges[:, 1]
    return xs, ys


def plot_contours(arr, aux, level=0.5, ax=None, **kwargs):
    if ax is None:
        _, ax = plt.subplots(1, 1, **kwargs)
    ax.imshow(arr, cmap=plt.cm.gray)
    xs, ys = find_edges(aux, level)
    ax.plot(xs, ys)


def crop_at_zyx_with_dhw(voxel, zyx, dhw, fill_with):
    '''Crop and pad on the fly.'''
    shape = voxel.shape
    # z, y, x = zyx
    # d, h, w = dhw
    crop_pos = []
    padding = [[0, 0], [0, 0], [0, 0]]
    for i, (center, length) in enumerate(zip(zyx, dhw)):
        assert length % 2 == 0
        # assert center < shape[i] # it's not necessary for "moved center"
        low = round(center) - length // 2
        high = round(center) + length // 2
        if low < 0:
            padding[i][0] = int(0 - low)
            low = 0
        if high > shape[i]:
            padding[i][1] = int(high - shape[i])
            high = shape[i]
        crop_pos.append([int(low), int(high)])
    cropped = voxel[crop_pos[0][0]:crop_pos[0][1], crop_pos[1]
                                                   [0]:crop_pos[1][1], crop_pos[2][0]:crop_pos[2][1]]
    if np.sum(padding) > 0:
        cropped = np.lib.pad(cropped, padding, 'constant',
                             constant_values=fill_with)
    return cropped


def window_clip(v, window_low=-1024, window_high=400, dtype=np.uint8):
    '''Use lung windown to map CT voxel to grey.'''
    # assert v.min() <= window_low
    return np.round(np.clip((v - window_low) / (window_high - window_low) * 255., 0, 255)).astype(dtype)


def resize(voxel, spacing, new_spacing=[1., 1., 1.]):
    '''Resize `voxel` from `spacing` to `new_spacing`.'''
    resize_factor = []
    for sp, nsp in zip(spacing, new_spacing):
        resize_factor.append(float(sp) / nsp)
    resized = scipy.ndimage.interpolation.zoom(voxel, resize_factor, mode='nearest')
    for i, (sp, shape, rshape) in enumerate(zip(spacing, voxel.shape, resized.shape)):
        new_spacing[i] = float(sp) * shape / rshape
    return resized, new_spacing


def rotation(array, angle):
    '''using Euler angles method.
    @author: renchao
    @params:
        angle: 0: no rotation, 1: rotate 90 deg, 2: rotate 180 deg, 3: rotate 270 deg
    '''
    #
    X = np.rot90(array, angle[0], axes=(0, 1))  # rotate in X-axis
    Y = np.rot90(X, angle[1], axes=(0, 2))  # rotate in Y'-axis
    Z = np.rot90(Y, angle[2], axes=(1, 2))  # rotate in Z"-axis
    return Z


def reflection(array, axis):
    '''
    @author: renchao
    @params:
        axis: -1: no flip, 0: Z-axis, 1: Y-axis, 2: X-axis
    '''
    if axis != -1:
        ref = np.flip(array, axis)
    else:
        ref = np.copy(array)
    return ref


def crop(array, zyx, dhw):
    z, y, x = zyx
    d, h, w = dhw
    cropped = array[z - d // 2:z + d // 2,
              y - h // 2:y + h // 2,
              x - w // 2:x + w // 2]
    return cropped


def random_center(shape, move):
    offset = np.random.randint(-move, move + 1, size=3)
    zyx = np.array(shape) // 2 + offset
    return zyx


def get_uniform_assign(length, subset):
    assert subset > 0
    per_length, remain = divmod(length, subset)
    total_set = np.random.permutation(list(range(subset)) * per_length)
    remain_set = np.random.permutation(list(range(subset)))[:remain]
    return list(total_set) + list(remain_set)


def split_validation(df, subset, by):
    df = df.copy()
    for sset in df[by].unique():
        length = (df[by] == sset).sum()
        df.loc[df[by] == sset, 'subset'] = get_uniform_assign(length, subset)
    df['subset'] = df['subset'].astype(int)
    return df


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse
_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)


# In[236]:


class Transform:

    def __init__(self, size, move):
        self.size = _triple(size)
        self.move = move

    def __call__(self, arr, aux=None):
        shape = arr.shape
        if self.move is not None:
            center = random_center(shape, self.move)
            # center = np.array(shape) // 2
            arr_ret = crop(arr, center, self.size)
            angle = np.random.randint(4, size=3)
            arr_ret = rotation(arr_ret, angle=angle)
            axis = np.random.randint(4) - 1
            arr_ret = reflection(arr_ret, axis=axis)
            arr_ret = np.expand_dims(arr_ret, axis=-1)
            if aux is not None:
                aux_ret = crop(aux, center, self.size)
                aux_ret = rotation(aux_ret, angle=angle)
                aux_ret = reflection(aux_ret, axis=axis)
                aux_ret = np.expand_dims(aux_ret, axis=-1)
                return arr_ret, aux_ret
            return arr_ret
        else:
            center = np.array(shape) // 2
            arr_ret = crop(arr, center, self.size)
            arr_ret = np.expand_dims(arr_ret, axis=-1)
            if aux is not None:
                aux_ret = crop(aux, center, self.size)
                aux_ret = np.expand_dims(aux_ret, axis=-1)
                return arr_ret, aux_ret
            return arr_ret


# In[237]:


class ClfDataset(Sequence):
    def __init__(self, crop_size=32, move=3):
        self.transform = Transform(crop_size,move)

    def __getitem__(self, item):
        name = df.loc[item, 'name']
        with np.load(os.path.join('work/机器学习/train_val', '%s.npz' % name)) as npz:
            voxel, seg = self.transform(npz['voxel'], npz['seg'])
        label = df.loc[item, 'lable']
        return voxel,  (label, seg)

    def __len__(self):
        return df.__len__()

    @staticmethod
    def _collate_fn(data):
        xs = []
        ys = []
        segs = []
        for x, y in data:
            xs.append(x)
            ys.append(y[0])
            segs.append(y[1])
        # return np.array([xs]).transpose(1, 2, 3, 4, 0), np.array(ys)
        return np.array(xs), {"clf": np.array(ys), "seg": np.array(segs)}

class ClfDataset_val(Sequence):
    def __init__(self, crop_size=32, move=3):
        self.transform = Transform(crop_size,move)

    def __getitem__(self, item):
        name = df.loc[item, 'name']
        with np.load(os.path.join('work/机器学习/train_val', '%s.npz' % name)) as npz:
            voxel, seg = self.transform(npz['voxel'], npz['seg'])
        label = df.loc[item, 'lable']
        return voxel,  (label, seg)

    def __len__(self):
        return df.__len__()

    @staticmethod
    def _collate_fn(data):
        xs = []
        ys = []
        segs = []
        for x, y in data:
            xs.append(x)
            ys.append(y[0])
            segs.append(y[1])
        # return np.array([xs]).transpose(1, 2, 3, 4, 0), np.array(ys)
        return np.array(xs), {"clf": np.array(ys), "seg": np.array(segs)}

class ClfDataset_test(Sequence):
    def __init__(self, crop_size=32, move=3):
        self.transform = Transform(crop_size,move)

    def __getitem__(self, item):
        name = result.loc[item, 'name']
        with np.load(os.path.join('test', '%s.npz' % name)) as npz:
            voxel, seg = self.transform(npz['voxel'], npz['seg'])
            # voxel = voxel*seg
        # label = df.loc[item, 'lable']
        return voxel

    def __len__(self):
        
        
        return df.__len__()

    @staticmethod
    def _collate_fn(data):
        xs = []
        for x in data:
            xs.append(x)
        # return np.array([xs]).transpose(1, 2, 3, 4, 0), np.array(ys)
        return np.array(xs)
        
        
def shuffle_iterator(iterator):
    # iterator should have limited size
    index = list(iterator)
    total_size = len(index)
    i = 0
    random.shuffle(index)
    while True:
        yield index[i]
        i += 1
        if i >= total_size:
            i = 0
            random.shuffle(index)

def get_loader_train(dataset, batch_size):
    total_size = 365
    print('Size', total_size)
    index_generator = shuffle_iterator(range(total_size))
    while True:
        data = []
        for _ in range(batch_size):
            idx = next(index_generator)
            # idx2 = next(index_generator)
            data.append(dataset[idx])
        yield dataset._collate_fn(data)



def get_loader_val(dataset, batch_size):
    total_size = 100
    print('Size', total_size)
    index_generator = shuffle_iterator(range(total_size))
    while True:
        data = []
        for _ in range(batch_size):
            idx = next(index_generator)
            data.append(dataset[idx+365])
        yield dataset._collate_fn(data)
        
def get_loader_test(dataset, batch_size):
    total_size = 117
    print('Size', total_size)
    # index_generator = shuffle_iterator(range(total_size))
    while True:
        data = []
        for i in range(total_size):
            # idx = next(index_generator)
            idx=i
            data.append(dataset[idx])
        yield dataset._collate_fn(data)


# In[238]:


dataset = ClfDataset(crop_size=36,move=3)
dataset_val = ClfDataset_val(crop_size=36,move=3)
test_dataset = ClfDataset_test(crop_size=32,move=5)
train_loader = get_loader_train(dataset, batch_size=50)
val_loader = get_loader_val(dataset_val, batch_size=50)
test_loader = get_loader_test(test_dataset, batch_size=117)

learning_rate=1.e-4
segmentation_task_ratio=0.1
weight_decay=0.
save_folder='test'
epochs=20
PARAMS = {
    'activation': lambda: Activation('relu'),  # the activation functions
    'bn_scale': True,  # whether to use the scale function in BN
    'weight_decay': 0.,  # l2 weight decay
    'kernel_initializer': 'he_uniform',  # initialization
    'first_scale': lambda x: x / 128. - 1.,  # the first pre-processing function
    'dhw': [32,32,32],  # the input shape
    'k': 16,  # the `growth rate` in DenseNet
    'bottleneck': 4,  # the `bottleneck` in DenseNet
    'compression': 2,  # the `compression` in DenseNet
    'first_layer': 32,  # the channel of the first layer
    'down_structure': [4, 4, 4],  # the down-sample structure
    'output_size': 1,  # the output number of the classification head
    'dropout_rate': None  # whether to use dropout, and how much to use
}


# In[239]:


from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.layers import (Conv3D, BatchNormalization, AveragePooling3D, concatenate, Lambda, SpatialDropout3D,
                          Activation, Input, GlobalAvgPool3D, Dense, Conv3DTranspose, add)
from keras.regularizers import l2 as l2_penalty
from keras.models import Model
import keras.backend as K


# In[240]:


def _conv_block(x, filters):
    bn_scale = PARAMS['bn_scale']
    activation = PARAMS['activation']
    kernel_initializer = PARAMS['kernel_initializer']
    weight_decay = PARAMS['weight_decay']
    bottleneck = PARAMS['bottleneck']
    dropout_rate = PARAMS['dropout_rate']

    x = BatchNormalization(scale=bn_scale, axis=-1)(x)
    x = activation()(x)
    x = Conv3D(filters * bottleneck, kernel_size=(1, 1, 1), padding='same', use_bias=False,
               kernel_initializer=kernel_initializer,
               kernel_regularizer=l2_penalty(weight_decay))(x)
    if dropout_rate is not None:
        x = SpatialDropout3D(dropout_rate)(x)
    x = BatchNormalization(scale=bn_scale, axis=-1)(x)
    x = activation()(x)
    x = Conv3D(filters, kernel_size=(3, 3, 3), padding='same', use_bias=True,
               kernel_initializer=kernel_initializer,
               kernel_regularizer=l2_penalty(weight_decay))(x)
    return x


def _dense_block(x, n):
    k = PARAMS['k']

    for _ in range(n):
        conv = _conv_block(x, k)
        x = concatenate([conv, x], axis=-1)
    return x


def _transmit_block(x, is_last):
    bn_scale = PARAMS['bn_scale']
    activation = PARAMS['activation']
    kernel_initializer = PARAMS['kernel_initializer']
    weight_decay = PARAMS['weight_decay']
    compression = PARAMS['compression']

    x = BatchNormalization(scale=bn_scale, axis=-1)(x)
    x = activation()(x)
    if is_last:
        x = GlobalAvgPool3D()(x)
    else:
        *_, f = x.get_shape().as_list()
        x = Conv3D(f // compression, kernel_size=(1, 1, 1), padding='same', use_bias=True,
                   kernel_initializer=kernel_initializer,
                   kernel_regularizer=l2_penalty(weight_decay))(x)
        x = AveragePooling3D((2, 2, 2), padding='valid')(x)
    return x


def get_model(weights=None, verbose=True, **kwargs):
    for k, v in kwargs.items():
        assert k in PARAMS
        PARAMS[k] = v
    if verbose:
        print("Model hyper-parameters:", PARAMS)

    dhw = PARAMS['dhw']
    first_scale = PARAMS['first_scale']
    first_layer = PARAMS['first_layer']
    kernel_initializer = PARAMS['kernel_initializer']
    weight_decay = PARAMS['weight_decay']
    down_structure = PARAMS['down_structure']
    output_size = PARAMS['output_size']

    shape = dhw + [1]

    inputs = Input(shape=shape)

    if first_scale is not None:
        scaled = Lambda(first_scale)(inputs)
    else:
        scaled = inputs
    conv = Conv3D(first_layer, kernel_size=(3, 3, 3), padding='same', use_bias=True,
                  kernel_initializer=kernel_initializer,
                  kernel_regularizer=l2_penalty(weight_decay))(scaled)

    downsample_times = len(down_structure)
    top_down = []
    for l, n in enumerate(down_structure):
        db = _dense_block(conv, n)
        top_down.append(db)
        conv = _transmit_block(db, l == downsample_times - 1)

    feat = top_down[-1]
    for top_feat in reversed(top_down[:-1]):
        *_, f = top_feat.get_shape().as_list()
        deconv = Conv3DTranspose(filters=f, kernel_size=2, strides=2, use_bias=True,
                                 kernel_initializer=kernel_initializer,
                                 kernel_regularizer=l2_penalty(weight_decay))(feat)
        feat = add([top_feat, deconv])
    seg_head = Conv3D(1, kernel_size=(1, 1, 1), padding='same',
                      activation='sigmoid', use_bias=True,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=l2_penalty(weight_decay),
                      name='seg')(feat)

    if output_size == 1:
        last_activation = 'sigmoid'
    else:
        last_activation = 'softmax'

    clf_head = Dense(output_size, activation=last_activation,
                     kernel_regularizer=l2_penalty(weight_decay),
                     kernel_initializer=kernel_initializer,
                     name='clf')(conv)

    model = Model(inputs, [clf_head, seg_head])
    if verbose:
        model.summary()

    if weights is not None:
        model.load_weights(weights)
    return model


# In[226]:


class DiceLoss:
    def __init__(self, beta=1., smooth=1.):
        self.__name__ = 'dice_loss_' + str(int(beta * 100))
        self.beta = beta  # the more beta, the more recall
        self.smooth = smooth

    def __call__(self, y_true, y_pred):
        bb = self.beta * self.beta
        y_true_f = K.batch_flatten(y_true)
        y_pred_f = K.batch_flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f, axis=-1)
        weighted_union = bb * K.sum(y_true_f, axis=-1) +                          K.sum(y_pred_f, axis=-1)
        score = -((1 + bb) * intersection + self.smooth) /                 (weighted_union + self.smooth)
        return score


# In[227]:


model = get_model('work/tmp8/test/weights.20.h5py')
model.compile(optimizer=Adam(lr=1.e-4), loss={"clf": 'binary_crossentropy', "seg": DiceLoss()},
              metrics={'clf': 'accuracy', 'seg': 'accuracy'}, loss_weights={"clf": 1., "seg": -0.1})
# checkpointer = ModelCheckpoint(filepath='work/tmp14/%s/weights.{epoch:02d}.h5py' % save_folder, verbose=1,
#                               period=1, save_weights_only=True)
# best_keeper = ModelCheckpoint(filepath='work/tmp14/%s/best.h5py' % save_folder, verbose=1, save_weights_only=True,
#                               monitor='val_clf_accuracy', save_best_only=True, period=1, mode='max')
# csv_logger = CSVLogger('work/tmp14/%s/training.csv' % save_folder)
# # early_stopping = EarlyStopping(monitor='val_clf_accuracy', min_delta=0, mode='max',
# #                                 patience=30, verbose=1)
# lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.334, patience=10,
#                               verbose=1, mode='min', epsilon=1.e-5, cooldown=2, min_lr=0)
# model.fit_generator(generator=train_loader, steps_per_epoch=20, max_queue_size=500, workers=1,
#                     validation_data=val_loader, epochs=epochs, validation_steps=15,
#                     callbacks=[checkpointer, best_keeper, lr_reducer, csv_logger])


# 对模型进行测试，保存测试结果到result.CSV

# In[228]:


test_data=next(test_loader)
print(test_data.shape)
b=model.predict(test_data)
result['Score']=b[0]
save=pd.DataFrame(data=result)
save.to_csv('work/机器学习/result.csv')


# In[229]:


print(b[0])


# In[ ]:




