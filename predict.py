from w_net_v11 import get_unet
from data_loader2 import get_data_generators
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
import glob
import os


img_rows, img_cols = 192, 672 / 2

w_net, disp_maps_forward = get_unet(img_rows=img_rows, img_cols=img_cols)
w_net.load_weights('weights/w_net_V12_epoch_14.h5')

in_dir = 'my_samples/'
out_dir = 'depth/'
data_generator, filenames = get_data_generators(in_dir,
                                            in_dir, 
                                            batch_size=1,
                                            shuffle=False, img_rows=img_rows, img_cols=img_cols)
in_files = [os.path.basename(f) for f in filenames]
for i in tqdm(range(len(in_files))):
    dat = data_generator.next()
    disparity_map_left, disparity_map_right = disp_maps_forward.predict(dat[0][0:10])

    depthMap_left = np.zeros(disparity_map_left[0,...,0].shape)
    for i_disp, disp in zip(range(-16,16),np.rollaxis(disparity_map_left[0,...],2)):
        depthMap_left += disp*i_disp

    scipy.misc.imsave(out_dir + in_files[i], depthMap_left)

