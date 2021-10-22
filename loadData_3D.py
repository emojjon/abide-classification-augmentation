import glob
import os
import sys
import numpy as np

import nibabel as nib
import tqdm

# from progress.bar import Bar

def load_data_3D(subfolder=''):

    dataset_path = subfolder
    if not os.path.isdir(dataset_path):
        print(' Dataset ' + subfolder + ' does not exist')

    # volume paths
    trainA_path = os.path.join(dataset_path, 'ASDS')
    trainB_path = os.path.join(dataset_path, 'CONTROLS')

    # volume file names
    trainA_volume_names = sorted(glob.glob(os.path.join(trainA_path,'*.nii*')))
    trainB_volume_names = sorted(glob.glob(os.path.join(trainB_path,'*.nii*')))

    trainA_volume_names = [os.path.basename(x) for x in trainA_volume_names]
    trainB_volume_names = [os.path.basename(x) for x in trainB_volume_names]

    # Examine one volume to get size and number of channels
    vol_test_A = nib.load(os.path.join(trainA_path, trainA_volume_names[0]))
    vol_test_B = nib.load(os.path.join(trainB_path, trainB_volume_names[0]))

    if len(vol_test_A.shape) == 3:
        volume_size_A = vol_test_A.shape
        nr_of_channels_A = 1
    else:
        volume_size_A = vol_test_A.shape[0:-1]
        nr_of_channels_A = vol_test_A.shape[-1]

    if len(vol_test_B.shape) == 3:
        volume_size_B = vol_test_B.shape
        nr_of_channels_B = 1
    else:
        volume_size_B = vol_test_B.shape[0:-1]
        nr_of_channels_B = vol_test_B.shape[-1]

    trainA_volumes = create_volume_array(trainA_volume_names, trainA_path, volume_size_A, nr_of_channels_A)
    trainB_volumes = create_volume_array(trainB_volume_names, trainB_path, volume_size_B, nr_of_channels_B)

    X = np.vstack((trainA_volumes, trainB_volumes))
    Y = np.zeros(X.shape[0])
    Y[:trainA_volumes.shape[0]] = 1

    # Normalize input volumes
    normConstant = np.quantile(X, 0.995)
    print('normConstant: {}'.format(normConstant))

    return {"volume_size": volume_size_A, "nr_of_channels": nr_of_channels_A,
            "X": X/normConstant, "Y": Y}

def create_volume_array(volume_list, volume_path, volume_size, nr_of_channels):
    #bar = Bar('Loading...', max=len(volume_list))

    # Define volume array
    volume_array = np.empty((len(volume_list),) + (volume_size) + (nr_of_channels,), dtype="float32")
    i = 0
    for volume_name in tqdm.tqdm(volume_list, desc = 'Loading...'):

        # Load volume and convert into np.array
        volume = nib.load(os.path.join(volume_path, volume_name)).get_fdata()  # Normalized to [0,1]
        volume = volume.astype("float32")

        # Add third dimension if volume is 2D
        if nr_of_channels == 1:  # Gray scale volume -> MR volume
            volume = volume[:, :, :, np.newaxis]

        # Add volume to array
        volume_array[i, :, :, :, :] = volume
        i += 1
        # bar.next()
    # bar.finish()

    return volume_array
