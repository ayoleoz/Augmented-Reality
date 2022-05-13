import numpy as np

def est_pixel_world(pixels, R_wc, t_wc, K):
    """
    Estimate the world coordinates of a point given a set of pixel coordinates.
    The points are assumed to lie on the x-y plane in the world.
    Input:
        pixels: N x 2 coordiantes of pixels
        R_wc: (3, 3) Rotation of camera in world
        t_wc: (3, ) translation from world to camera
        K: 3 x 3 camara intrinsics
    Returns:
        Pw: N x 3 points, the world coordinates of pixels
    """

    ##### STUDENT CODE START #####
    newpixel = np.concatenate((pixels, np.ones(shape = (len(pixels), 1))), axis=1)
    Pc = np.dot(R_wc, np.dot(np.linalg.inv(K), newpixel.T))
    lambdaa = (- t_wc[2] / Pc[2]).reshape(-1)
    Pw = (t_wc.reshape(3,1) + lambdaa * Pc).T
    ##### STUDENT CODE END #####
    return Pw
