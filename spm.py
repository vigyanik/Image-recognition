from utils import load_cifar10_data
from utils import extract_DenseSift_descriptors
from utils import build_codebook
from utils import input_vector_encoder
from classifier import svm_classifier

import numpy as np


#def spatial_pyramid_matching(image, keypoints, descriptors, codebook, level):
#    """
#    Spatial Pyramid Matching
#    descriptors: Usually using dense SIFT
#    keypoints: location of these descriptors
#    level: pyramid level,int, support 1 or 2 or 3
#    if level==1, it means bag of feature
#    if codebook size is 150,
#    will return a feature vector of size (1,3150)
#    """
#    # Scaling this image to 16 cells
#    #  a0  a1  a2  a3
#    #  a4  a5  a6  a7
#    #  a8  a9  a10 a11
#    #  a12 a13 a14 a15
#    # e.g.
#    # img.shape = (200,299)
#    # img[:50,:74] is the top left corner
#    # img[50*3:,74*3:] is right bottom corner
#    height, width, channel = image.shape
#    height_step = height / 4.0
#    width_step = width / 4.0
#    crop = [[] for i in xrange(16)]
#    for idx,kp in enumerate(keypoints):
#        if kp.pt[1] <= height_step: # row 0
#            if 0 <= kp.pt[0] <= width_step: # img[0][0]
#                crop[0].append(descriptors[idx,:])
#            if width_step < kp.pt[0] <= 2 * width_step: # img[0][1]
#                crop[1].append(descriptors[idx,:])
#            if 2 * width_step < kp.pt[0] <= 3 * width_step: # img[0][2]
#                crop[2].append(descriptors[idx,:])
#            if 3 * width_step < kp.pt[0] <= 4 * width_step: # img[0][3]
#                crop[3].append(descriptors[idx,:])
#        if height_step < kp.pt[1] <= 2 * height_step: # row1
#            if 0 <= kp.pt[0] <= width_step: # img[1][0]
#                crop[4].append(descriptors[idx,:])
#            if width_step < kp.pt[0] <= 2 * width_step: # img[1][1]
#                crop[5].append(descriptors[idx,:])
#            if 2 * width_step < kp.pt[0] <= 3 * width_step: # img[1][2]
#                crop[6].append(descriptors[idx,:])
#            if 3 * width_step < kp.pt[0] <= 4 * width_step: # img[1][3]
#                crop[7].append(descriptors[idx,:])
#        if 2 * height_step < kp.pt[1] <= 3 * height_step: # row2
#            if 0 <= kp.pt[0] <= width_step: # img[2][0]
#                crop[8].append(descriptors[idx,:])
#            if width_step < kp.pt[0] <= 2 * width_step: # img[2][1]
#                crop[9].append(descriptors[idx,:])
#            if 2 * width_step < kp.pt[0] <= 3 * width_step: # img[2][2]
#                crop[10].append(descriptors[idx,:])
#            if 3 * width_step < kp.pt[0] <= 4 * width_step: # img[2][3]
#                crop[11].append(descriptors[idx,:])
#        if 3 * height_step < kp.pt[1] <= 4 * height_step: # row3
#            if 0 <= kp.pt[0] <= width_step: # img[3][0]
#                crop[12].append(descriptors[idx,:])
#            if width_step < kp.pt[0] <= 2 * width_step: # img[3][1]
#                crop[13].append(descriptors[idx,:])
#            if 2 * width_step < kp.pt[0] <= 3 * width_step: # img[3][2]
#                crop[14].append(descriptors[idx,:])
#            if 3 * width_step < kp.pt[0] <= 4 * width_step: # img[3][3]
#                crop[15].append(descriptors[idx,:])
#    level3 = [input_vector_encoder(np.asarray(crop[i]),codebook) for i in range(16)]
#    hist_of_level3 = np.asarray(level3).flatten()
#
#    # for building level 2
#    crop2 = [[],[],[],[]]
#    crop2[0] = crop[0] + crop[1] + crop[4] + crop[5] #img[0][0]
#    crop2[1] = crop[2] + crop[3] + crop[6] + crop[7] #img[0][1]
#    crop2[2] = crop[8] + crop[9] + crop[12] + crop[13] #img[1][0]
#    crop2[3] = crop[10] + crop[11] + crop[14] + crop[15] #img[1][1]
#    level2 = [input_vector_encoder(np.asarray(crop2[i]),codebook) for i in range(4)]
#    hist_of_level2 = np.asarray(level2).flatten()
#
#    # for building level 1, i.e. bag of feature
#    hist_of_level1 = input_vector_encoder(descriptors,codebook)
#
#    if level == 1:
#        return hist_of_level1
#    if level == 2:
#        l1 = hist_of_level1 * 0.5
#        l2 = hist_of_level2 * 0.5
#        return np.concatenate((l1,l2))
#    if level == 3:
#        l1 = hist_of_level1 * 0.25
#        l2 = hist_of_level2 * 0.25
#        l3 = hist_of_level3 * 0.5
#        return np.concatenate((l1,l2,l3))




def build_spatial_pyramid(image, descriptor, level):
    """
    Rebuild the descriptors according to the level of pyramid
    """
    assert 0 <= level <= 2, "Level Error"
    step_size = DSIFT_STEP_SIZE
    from utils import DSIFT_STEP_SIZE as s
    assert s == step_size, "step_size must equal to DSIFT_STEP_SIZE\
                            in utils.extract_DenseSift_descriptors()"
    h = image.shape[0] / step_size
    w = image.shape[1] / step_size
    idx_crop = np.array(range(len(descriptor))).reshape(h,w)
    size = idx_crop.itemsize
    height, width = idx_crop.shape
    bh, bw = 2**(3-level), 2**(3-level)
    shape = (height/bh, width/bw, bh, bw)
    strides = size * np.array([width*bh, bw, width, 1])
    crops = np.lib.stride_tricks.as_strided(
            idx_crop, shape=shape, strides=strides)
    des_idxs = [col_block.flatten().tolist() for row_block in crops
                for col_block in row_block]
    pyramid = []
    for idxs in des_idxs:
        pyramid.append(np.asarray([descriptor[idx] for idx in idxs]))
    return pyramid

def spatial_pyramid_matching(image, descriptor, codebook, level):
    pyramid = []
    if level == 0:
        pyramid += build_spatial_pyramid(image, descriptor, level=0)
        code = [input_vector_encoder(crop, codebook) for crop in pyramid]
        return np.asarray(code).flatten()
    if level == 1:
        pyramid += build_spatial_pyramid(image, descriptor, level=0)
        pyramid += build_spatial_pyramid(image, descriptor, level=1)
        code = [input_vector_encoder(crop, codebook) for crop in pyramid]
        code_level_0 = 0.5 * np.asarray(code[0]).flatten()
        code_level_1 = 0.5 * np.asarray(code[1:]).flatten()
        return np.concatenate((code_level_0, code_level_1))
    if level == 2:
        pyramid += build_spatial_pyramid(image, descriptor, level=0)
        pyramid += build_spatial_pyramid(image, descriptor, level=1)
        pyramid += build_spatial_pyramid(image, descriptor, level=2)
        code = [input_vector_encoder(crop, codebook) for crop in pyramid]
        code_level_0 = 0.25 * np.asarray(code[0]).flatten()
        code_level_1 = 0.25 * np.asarray(code[1:5]).flatten()
        code_level_2 = 0.5 * np.asarray(code[5:]).flatten()
        return np.concatenate((code_level_0, code_level_1, code_level_2))


VOC_SIZE = 100
PYRAMID_LEVEL = 1

DSIFT_STEP_SIZE = 4
# DSIFT_STEP_SIZE is related to the function
# extract_DenseSift_descriptors in utils.py
# and build_spatial_pyramid in spm.py


if __name__ == '__main__':

    x_train, y_train = load_cifar10_data(dataset='train')
    x_test, y_test = load_cifar10_data(dataset='test')

    print "Dense SIFT feature extraction"
    x_train_feature = [extract_DenseSift_descriptors(img) for img in x_train]
    x_test_feature = [extract_DenseSift_descriptors(img) for img in x_test]
    x_train_kp, x_train_des = zip(*x_train_feature)
    x_test_kp, x_test_des = zip(*x_test_feature)

    print "Train/Test split: {:d}/{:d}".format(len(y_train), len(y_test))
    print "Codebook Size: {:d}".format(VOC_SIZE)
    print "Pyramid level: {:d}".format(PYRAMID_LEVEL)
    print "Building the codebook, it will take some time"
    codebook = build_codebook(x_train_des, VOC_SIZE)
    import cPickle
    with open('./spm_lv1_codebook.pkl','w') as f:
        cPickle.dump(codebook, f)

    print "Spatial Pyramid Matching encoding"
    x_train = [spatial_pyramid_matching(x_train[i],
                                        x_train_des[i],
                                        codebook,
                                        level=PYRAMID_LEVEL)
                                        for i in xrange(len(x_train))]

    x_test = [spatial_pyramid_matching(x_test[i],
                                       x_test_des[i],
                                       codebook,
                                       level=PYRAMID_LEVEL) for i in xrange(len(x_test))]

    x_train = np.asarray(x_train)
    x_test = np.asarray(x_test)

    svm_classifier(x_train, y_train, x_test, y_test)
