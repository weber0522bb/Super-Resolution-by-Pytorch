import numpy as np

def calc_psnr(img_x, img_y, max_val=1.0,border_ignore=4):
    if border_ignore <= 0:
        se = np.square(img_x - img_y)
    else:
        se = np.square(img_x[border_ignore:-1-border_ignore+1,border_ignore:-1-border_ignore+1,:] - img_y[border_ignore:-1-border_ignore+1,border_ignore:-1-border_ignore+1,:])
    mse = np.mean(se)
    psnr = 10*np.log10(max_val*max_val/mse)
    return psnr

