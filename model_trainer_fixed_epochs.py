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
no_cv_splits = 10
batch_size = 16
no_of_epochs = 150
early_stopping = False
patience = 50
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

# Split data
if len(sys.argv) > 1:
    fold = int(sys.argv[1])
else:
    fold = 0

X = data['X']
Y = data['Y']
# X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size = 0.3, random_state=seed)
# X_val, X_test, Y_val, Y_test = train_test_split(X_val, Y_val, test_size = 0.5, random_state=seed)

X, X_test, Y, Y_test = train_test_split(X, Y, test_size = 0.15, random_state=seed)

kf = KFold(n_splits=no_cv_splits)
indices = [(I_train,I_test) for I_train, I_test in kf.split(X)]

X_train = X[indices[fold][0]]
Y_train = Y[indices[fold][0]]
X_val = X[indices[fold][1]]
Y_val = Y[indices[fold][1]]

# Prepare results dir
mod_time = f'{time.strftime("%Y-%m-%d_%H.%M.%S")}'
if 'SPEC' in os.environ:
    mod_spec = os.environ['SPEC']
else:
    mod_spec = 'unknown'

mid = '{}_i{}_{}'.format(mod_spec, fold, mod_time)

model_name = 'model_' + mid

for d in ('RESULTS', os.path.join('RESULTS', mid)):
    if not os.path.isdir(os.path.join(base_dir,d)):
        os.mkdir(os.path.join(base_dir,d))

res_path = os.path.join(base_dir, 'RESULTS', mid)

## Model

vol_size = ((data['volume_size']) + (data['nr_of_channels'],))

# Create model
model = models.Sequential(name = os.path.join(res_path, model_name))
model.add(layers.Conv3D(8, (3, 3, 3), input_shape = vol_size, kernel_initializer='he_normal'))
# model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling3D((2, 2, 2)))
model.add(layers.Conv3D(16, (3, 3, 3), kernel_initializer = 'he_normal'))
# model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling3D((2, 2, 2)))
model.add(layers.Conv3D(16, (3, 3, 3), kernel_initializer = 'he_normal'))
# model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))

model.add(layers.Flatten())

model.add(layers.Dense(16, activation='relu'))
# model.add(layers.GlobalAveragePooling3D())
# model.add(layers.Dropout(0.2))
model.add(layers.Dense(1, activation='sigmoid'))

optim = optimizers.Adam(learning_rate=1e-5)
# optim = optimizers.SGD(learning_rate=0.01, momentum=0.0)
model.compile(optimizer=optim,
              loss='bce',
              metrics=['accuracy'])

# Save to report
with open(os.path.join(res_path, f'report_{mid}.txt'), "w") as report:
    print(f'Running:\n\t{sys.argv[0]}\n', file = report)
    print('Commandline Parameters:', file = report)
    for i in range(1, len(sys.argv)):
        print(f'\t{i}: "{sys.argv[i]}"', file = report)
    # print(f"This model uses a per derivative normalisation factor (memoised in .gamma_norm).", file = report)
    # print(f"Trained model on ABIDE/{der}", file = report)
    model.summary(print_fn = lambda s: print(s, file = report))

# Save model plot
from tensorflow.keras.utils import plot_model
# plot_model(model, to_file=os.path.join(res_path, f'graph_{mid}.png'),
#            show_shapes=True, show_dtype=True, expand_nested=True, dpi=600)


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

## Training

rng = np.random.default_rng()

tr_set_sz = X_train.shape[0]
no_of_steps = math.ceil(tr_set_sz / batch_size)

metrics_train = np.empty((no_of_epochs, no_of_steps, len(model.test_on_batch(X_train[0 : batch_size], Y_train[0 : batch_size]))))
no_of_metrics = metrics_train.shape[2]
metrics_names = model.metrics_names + ['val_' + n for n in model.metrics_names]
history = lambda : None
history.__dict__['history'] = {k : np.empty((no_of_epochs, )) for k in metrics_names}
history.__dict__['epoch'] = []

best_val_acc = -math.inf
best_val_acc_e = 0
no_char_epochs = int(math.log(no_of_epochs, 10)) + 1
asportatio_interruptus = False

for e in range(no_of_epochs):
    index = np.random.permutation(tr_set_sz)

    # Train on a batch
    for b in tqdm.trange(no_of_steps, desc = '    Step', leave=False):
        if (b + 1) * batch_size < tr_set_sz:
            Xb = X_train[index[b * batch_size : (b + 1) * batch_size], :, :, :]
            Yb = Y_train[index[b * batch_size : (b + 1) * batch_size]]
        else:
            Xb = X_train[index[b * batch_size : ], :, :, :]
            Yb = Y_train[index[b * batch_size : ]]

        Xc = aug_batch(Xb)
        metrics_train[e, b, :] = model.train_on_batch(Xb, Yb)

    # Store metrics
    metrics_val = model.evaluate(X_val, Y_val, verbose = 0)
    for i, m in enumerate(model.metrics_names):
        history.history[m][e] = metrics_train[e, :, i].mean()
        history.history['val_' + m][e] = metrics_val[i]

    # Save best model
    if history.history['val_accuracy'][e] > best_val_acc:
        best_val_acc = history.history['val_accuracy'][e]
        best_val_acc_e = e
        tf.keras.models.save_model(model,
                                   os.path.join(res_path, f'model_{mid}'),
                                   save_format='tf')
    history.epoch.append(e)

    # Print metrics
    print(f"{e+1:>{no_char_epochs}}/{no_of_epochs:>{no_char_epochs}}: ", end='')
    for n in metrics_names:
        print(f"{n}: {history.history[n][e]:.4f}   ", end = '')
    print()

    # Early stopping
    if early_stopping and e - best_val_acc_e > patience:
        asportatio_interruptus = True
        print(f"Stopping training because validation accuracy has not improved for {patience} epochs.")
        break

## Results

aug_batch.pool.close()

# Early stopping
if asportatio_interruptus:
    # Load best model
    model = tf.keras.models.load_model(os.path.join(res_path, f'model_{mid}'))

    # Remove history after best epoch
    history.epoch = history.epoch[0 : best_val_acc_e + 1]
    for k in history.history:
        history.history[k] = history.history[k][0 : best_val_acc_e + 1]

    # Rename saved model path
    os.rename(os.path.join(res_path, f'model_{mid}'),
              os.path.join(res_path, f'model_{mid}_{e}_F'))
else:
    # Save final model
    tf.keras.models.save_model(model,
                               os.path.join(res_path, f'model_{mid}_{e}_F'),
                               save_format='tf')

# Save training metrics
np.save(os.path.join(res_path, f'training_metrics_{mid}.npy'), metrics_train)

# Save test metrics for final epoch
test_loss = tf.keras.losses.BinaryCrossentropy(reduction = tf.keras.losses.Reduction.NONE)
Y_pred = model.predict(X_test)
sample_loss = test_loss(Y_test, Y_pred)

test_stats_final = np.empty((X_test.shape[0], 3))
test_stats_final[:, 0] = Y_test
test_stats_final[:, 1] = Y_pred.squeeze()
test_stats_final[:, 2] = sample_loss.numpy()
np.save(os.path.join(res_path, f'test_stats_{mid}_F.npy'), test_stats_final)

# Save test metrics for best validation epoch
model = tf.keras.models.load_model(os.path.join(res_path, f'model_{mid}'))

test_loss = tf.keras.losses.BinaryCrossentropy(reduction = tf.keras.losses.Reduction.NONE)
Y_pred = model.predict(X_test)
sample_loss = test_loss(Y_test, Y_pred)

test_stats_best = np.empty((X_test.shape[0], 3))
test_stats_best[:, 0] = Y_test
test_stats_best[:, 1] = Y_pred.squeeze()
test_stats_best[:, 2] = sample_loss.numpy()
np.save(os.path.join(res_path, f'test_stats_{mid}.npy'), test_stats_best)

# Save results in report
h = history
with open(os.path.join(res_path, f'report_{mid}.txt'), "a") as report:
    # Number of epochs trained
    if asportatio_interruptus:
        print(f"Stopped after {len(h.epoch)} epochs because of the validation accuracy not improving.", file = report)
    else:
        print(f"Trained {no_of_epochs} epochs", file = report)

    # Average test metrics
    print("Final metrics:", file = report)
    correct_tests = test_stats_final[abs(test_stats_final[:, 0] - test_stats_final[:, 1]) < 0.5]
    print(f"Test accuracy: {correct_tests.shape[0] / test_stats_final.shape[0]:.4f} \n"
          f"Test loss: {test_stats_final[:, 2].mean():.4f}", file = report)

    print("Best epoch metrics:", file = report)
    correct_tests = test_stats_best[abs(test_stats_best[:, 0] - test_stats_best[:, 1]) < 0.5]
    print(f"Test accuracy: {correct_tests.shape[0] / test_stats_best.shape[0]:.4f} \n"
          f"Test loss: {test_stats_best[:, 2].mean():.4f}", file = report)

    # Training / validation highlights
    for k, v in h.history.items():
        imax = np.argmax(v)
        imin = np.argmin(v)
        print(f'{k}:\t{v[-1]:.5f}\t(maximum value {v[imax]:.5f} after epoch {imax}\tand minimum value {v[imin]:.5f} after epoch {imin})', file = report)

    # Full training / validation metrics
    print("Complete metrics:", file = report)
    for k, v in h.history.items():
        print(f"{k}:", file = report)
        for n in v:
            print(f"{n}", file = report)

# Save training curves
def save_plots(history, path):
    plt.figure(figsize=(10,4))

    plt.subplot(211)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['Training', 'Validation'])

    plt.subplot(212)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.legend(['Training', 'Validation'])

    plt.savefig(path, dpi=200)

fig_path = os.path.join(res_path, f'metrics_{mid}.png')
save_plots(history, fig_path)
