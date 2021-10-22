import numpy as np
from scipy.ndimage.interpolation import affine_transform
import elasticdeform
import multiprocessing as mp
import os
import sys

def patch_extraction(Xb, yb, sizePatches=128, Npatches=1):
    """
    3D patch extraction
    """

    batch_size, rows, columns, slices, channels = Xb.shape
    X_patches = np.empty((batch_size*Npatches, sizePatches, sizePatches, sizePatches, channels))
    y_patches = np.empty((batch_size*Npatches, sizePatches, sizePatches, sizePatches))
    i = 0
    for b in range(batch_size):
        for p in range(Npatches):
            x = np.random.randint(rows-sizePatches+1)
            y = np.random.randint(columns-sizePatches+1)
            z = np.random.randint(slices-sizePatches+1)

            X_patches[i] = Xb[b, x:x+sizePatches, y:y+sizePatches, z:z+sizePatches, :]
            y_patches[i] = yb[b, x:x+sizePatches, y:y+sizePatches, z:z+sizePatches]
            i += 1

    return X_patches, y_patches

def flip3D(X):
    """
    Flip the 3D image respect one of the 3 axis chosen randomly
    """
#     choice = np.random.randint(3)
#     if choice == 0: # flip on x
#         X_flip = X[::-1, :, :, :]
#     elif choice == 1: # flip on y
#         X_flip = X[:, ::-1, :, :]
#     else: # flip on z
#         X_flip = X[:, :, ::-1, :]

    X_flip = X[::-1, :, :, :]

    return X_flip


def rotation_zoom3D(X):
    """
    Rotate a 3D image with alfa, beta and gamma degree respect the axis x, y and z respectively.
    The three angles are chosen randomly between 0-30 degrees
    """
    if not ('ROTATION' in os.environ and os.environ['ROTATION']) and not ('SCALE' in os.environ and os.environ['SCALE']):
        return X
    if 'ROTATION' in os.environ and os.environ['ROTATION']:
        frac = int(os.environ['ROTATION'])
        if frac not in [12, 6, 3, 2]:
            sys.exit('Wrong rotation')
        alpha, beta, gamma = (np.random.random_sample(3) - 0.5) * np.pi/frac

        Rx = np.array([[1, 0, 0, 0],
                       [0, np.cos(alpha), -np.sin(alpha), 0],
                       [0, np.sin(alpha), np.cos(alpha), 0],
                       [0, 0, 0, 1]])

        Ry = np.array([[np.cos(beta), 0, np.sin(beta), 0],
                       [0, 1, 0, 0],
                       [-np.sin(beta), 0, np.cos(beta), 0],
                       [0, 0, 0, 1]])

        Rz = np.array([[np.cos(gamma), -np.sin(gamma), 0, 0],
                       [np.sin(gamma), np.cos(gamma), 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]])

        R_rot = Rx @ Ry @ Rz
    else:
        R_rot = np.eye(4)
    if 'SCALE' in os.environ and os.environ['SCALE']:

        sc = int(os.environ['SCALE'])
        if sc not in [1, 2]:
            sys.exit('Wrong scale')
#         a, b = 0.8, 1.2
        a, b = 1-sc/10, 1+sc/10

        alpha, beta, gamma = (b-a)*np.random.random_sample(3) + a

        R_scale = np.array([[alpha, 0, 0, 0],
                       [0, beta, 0, 0],
                       [0, 0, gamma, 0],
                       [0, 0, 0, 1]])
    else:
        R_scale = np.eye(4)


    # Need translation to rotate around center of volume!
    x, y, z = X.shape[0:3]
    Rt = np.array([[1, 0, 0, x/2],[0, 1, 0, y/2],[0, 0, 1, z/2],[0, 0, 0, 1]])
    Rt2 = np.array([[1, 0, 0, -x/2],[0, 1, 0, -y/2],[0, 0, 1, -z/2],[0, 0, 0, 1]])

    R = Rt @ R_rot @ R_scale @ Rt2
    X_rot = np.empty_like(X)
    for channel in range(X.shape[-1]):
        X_rot[:,:,:,channel] = affine_transform(X[:,:,:,channel], R, mode='constant')

    return X_rot
def brightness(X):
    """
    Changing the brighness of a image using power-law gamma transformation.
    Gain and gamma are chosen randomly for each image channel.

    Gain chosen between [0.8 - 1.2]
    Gamma chosen between [0.8 - 1.2]

    new_im = gain * im^gamma
    """

    X_new = np.zeros(X.shape)
    for c in range(X.shape[-1]):
        im = X[:,:,:,c]
        gain, gamma = (1.2 - 0.8) * np.random.random_sample(2,) + 0.8
        im_new = np.sign(im)*gain*(np.abs(im)**gamma)
        X_new[:,:,:,c] = im_new

    return X_new

def elastic(X, sigma=2):
    """::-1, :
    Elastic deformation on a image and its target
    """

    [Xel] = elasticdeform.deform_random_grid([X], sigma=sigma, axis=[(0, 1, 2), ], mode='constant')

    return Xel

def random_decisions(N):
    """
    Generate N random decisions for augmentation
    N should be equal to the batch size
    """

#     decisions = np.zeros((N, 4))  # 4 is number of aug techniques to combine (patch extraction excluded)
#     for n in range(N):
#         decisions[n] = np.random.randint(2, size=4)

    decisions = np.random.randint(2, size=(N, 4)) # 4 is number of aug techniques to /combine (patch extraction excluded)
#     decisions = np.ones((N, 4)) # 4 is number of aug techniques to combine (patch extraction excluded)

    return decisions

def combine_aug(X, do):
    """
    Combine randomly the different augmentation techniques written above
    """

    # Essential to reseed for multiprocessing. Otherwise all instances will get the same random numbers
    np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))

    Xnew = X

    # make sure to use at least 25% of original images
    if np.random.random_sample() > 0.75:
        return Xnew
    else:
        if do[0] and 'FLIP3D' in os.environ and os.environ['FLIP3D']:
            Xnew = flip3D(Xnew)

        if do[1] and 'BRIGHTNESS' in os.environ and os.environ['BRIGHTNESS']:
            Xnew = brightness(Xnew)

        if do[2]:
            Xnew = rotation_zoom3D(Xnew)

        if do[3] and 'ELASTIC' in os.environ and os.environ['ELASTIC']:
            sigma = int(os.environ['ELASTIC'])
            if sigma not in [2,4,6,8]:
                sys.exit('Wrong sigma')
            Xnew = elastic(Xnew, sigma)

        return Xnew

def aug_batch(Xb):
    """
    Generate a augmented image batch
    """
    no_channel_dim = Xb.ndim < 5

    if no_channel_dim: Xb = np.expand_dims(Xb, axis=-1)
    batch_size = len(Xb)
    newXb = np.empty_like(Xb)

    decisions = random_decisions(batch_size)

    inputs = [(X, do) for X, do in zip(Xb, decisions)]
    if not 'pool' in aug_batch.__dict__:
        aug_batch.pool = mp.Pool(processes=min(batch_size,32))
    multi_result = aug_batch.pool.starmap(combine_aug, inputs)
    # pool.close()

    for i in range(batch_size):
        newXb[i] = multi_result[i]

    if no_channel_dim:
        newXb = np.squeeze(newXb, axis=-1)

    return newXb
