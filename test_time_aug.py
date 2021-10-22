#!/usr/bin/env python3

# std lib
import math
import os
import sys
import time

# common stable modules
import matplotlib.pyplot as plt
import numpy as np
import tqdm

# common buggy modules
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from sklearn.model_selection import train_test_split, KFold

# our own code
import lrdebug
from loadData_3D import load_data_3D
from xaugmentation import aug_batch


# Parameters
seed = 101 #Random seed

# Select GPU
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
if len(sys.argv) > 2:
    os.environ["CUDA_VISIBLE_DEVICES"]=sys.argv[2]
else:
    os.environ["CUDA_VISIBLE_DEVICES"]="1"

# Set GPU memory growth
physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

## Data

# Load data
# if len(sys.argv) > 1:
#     base_dir = sys.argv[1]
# else:
#     base_dir = './data/reho'
base_dir = './data/reho'
data = load_data_3D(base_dir)

X = data['X']
Y = data['Y']

_, X_test, _, Y_test = train_test_split(X, Y, test_size = 0.15, random_state=seed)
n_test = Y_test.shape[0]

# Prepare results dir
mod_time = f'{time.strftime("%Y-%m-%d_%H.%M.%S")}'
if 'SPEC' in os.environ:
    mod_spec = os.environ['SPEC']
else:
    mod_spec = 'unknown'

## Model

# Load model
if len(sys.argv) > 1:
    model_path = sys.argv[1].rstrip('/')
else:
    # model_path = './data/reho/RESULTS/ba_i0_2021-10-12_10.23.41/model_ba_i0_2021-10-12_10.23.41_103_F'
    model_path = './data/reho/RESULTS/s10_i0_2021-10-10_21.50.21/model_s10_i0_2021-10-10_21.50.21_111_F'

res_path, model_name = os.path.split(model_path)
spec = model_name.split(sep='_')[1]

model = tf.keras.models.load_model(model_path)

##

def plot_stuff(a,b,i):
    plt.figure()
    plt.subplot(121)
    plt.imshow(a[i,:,:,30,0])
    plt.colorbar()
    plt.subplot(122)
    plt.imshow(b[i,:,:,30,0])
    plt.colorbar()
    plt.show()

## Test accuracy

# aug_batch.pool.close()

# Predict without TTA
Y_pred = model.predict(X_test).squeeze()
test_acc = ((Y_pred > 0.5) == Y_test).sum() / n_test

# Predict with TTA
Y_pred_tta = np.zeros_like(Y_test)
n_aug = 32

for i in tqdm.trange(n_test):
    Xb = np.repeat(X_test[[i],],n_aug,axis=0)
    Xc = aug_batch(Xb)

    res = model.predict_on_batch(Xc).squeeze()
    Y_pred_tta[i] = res.mean()
test_acc_tta = ((Y_pred_tta > 0.5) == Y_test).sum() / n_test

## Save results
with open(os.path.join(res_path, f'test_acc.txt'), "a") as report:
    print(f'Test accuracy: {test_acc:.4f}', file=report)
    print(f'Test accuracy TTA: {test_acc_tta:.4f}', file=report)

# ##
#
# # Save test metrics for final epoch
# test_loss = tf.keras.losses.BinaryCrossentropy(reduction = tf.keras.losses.Reduction.NONE)
# Y_pred = model.predict(X_test)
# sample_loss = test_loss(Y_test, Y_pred)
#
# test_stats_final = np.empty((X_test.shape[0], 3))
# test_stats_final[:, 0] = Y_test
# test_stats_final[:, 1] = Y_pred.squeeze()
# test_stats_final[:, 2] = sample_loss.numpy()
# np.save(os.path.join(res_path, f'test_stats_{mid}_F.npy'), test_stats_final)
#
# # Save test metrics for best validation epoch
# model = tf.keras.models.load_model(os.path.join(res_path, f'model_{mid}'))
#
# test_loss = tf.keras.losses.BinaryCrossentropy(reduction = tf.keras.losses.Reduction.NONE)
# Y_pred = model.predict(X_test)
# sample_loss = test_loss(Y_test, Y_pred)
#
# test_stats_best = np.empty((X_test.shape[0], 3))
# test_stats_best[:, 0] = Y_test
# test_stats_best[:, 1] = Y_pred.squeeze()
# test_stats_best[:, 2] = sample_loss.numpy()
# np.save(os.path.join(res_path, f'test_stats_{mid}.npy'), test_stats_best)
#
# # Save results in report
# h = history
# with open(os.path.join(res_path, f'report_{mid}.txt'), "a") as report:
#     # Number of epochs trained
#     if asportatio_interruptus:
#         print(f"Stopped after {len(h.epoch)} epochs because of the validation accuracy not improving.", file = report)
#     else:
#         print(f"Trained {no_of_epochs} epochs", file = report)
#
#     # Average test metrics
#     print("Final metrics:", file = report)
#     correct_tests = test_stats_final[abs(test_stats_final[:, 0] - test_stats_final[:, 1]) < 0.5]
#     print(f"Test accuracy: {correct_tests.shape[0] / test_stats_final.shape[0]:.4f} \n"
#           f"Test loss: {test_stats_final[:, 2].mean():.4f}", file = report)
#
#     print("Best epoch metrics:", file = report)
#     correct_tests = test_stats_best[abs(test_stats_best[:, 0] - test_stats_best[:, 1]) < 0.5]
#     print(f"Test accuracy: {correct_tests.shape[0] / test_stats_best.shape[0]:.4f} \n"
#           f"Test loss: {test_stats_best[:, 2].mean():.4f}", file = report)
#
#     # Training / validation highlights
#     for k, v in h.history.items():
#         imax = np.argmax(v)
#         imin = np.argmin(v)
#         print(f'{k}:\t{v[-1]:.5f}\t(maximum value {v[imax]:.5f} after epoch {imax}\tand minimum value {v[imin]:.5f} after epoch {imin})', file = report)
#
#     # Full training / validation metrics
#     print("Complete metrics:", file = report)
#     for k, v in h.history.items():
#         print(f"{k}:", file = report)
#         for n in v:
#             print(f"{n}", file = report)

