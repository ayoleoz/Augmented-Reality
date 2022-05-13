from est_homography import est_homography
import numpy as np

def PnP(Pc, Pw, K=np.eye(3)):
    """
    Solve Perspective-N-Point problem with collineation assumption, given correspondence and intrinsic

    Input:
        Pc: 4x2 numpy array of pixel coordinate of the April tag corners in (x,y) format
        Pw: 4x3 numpy array of world coordinate of the April tag corners in (x,y,z) format
    Returns:
        R: 3x3 numpy array describing camera orientation in the world (R_wc)
        t: (3, ) numpy array describing camera translation in the world (t_wc)

    """

    ##### STUDENT CODE START #####

    # Homography Approach
    # Following slides: Pose from Projective Transformation
    H = est_homography(Pw[:, :2], Pc)
    KinvH = np.linalg.inv(K).dot(H)
    h12 = np.cross(KinvH[:,0], KinvH[:,1])
    Hp = np.concatenate((KinvH[:,0].reshape((3,1)), KinvH[:,1].reshape((3,1)), h12.reshape((3,1))), axis = 1)

    U, S, V = np.linalg.svd(Hp)

    S = np.eye(3)
    S[2][2] = np.linalg.det(U@V)
    R = U@S@V
    R = np.transpose(R)
    t = KinvH[:,2] / np.linalg.norm(KinvH[:,0])
    t = - R@t
    ##### STUDENT CODE END #####

    return R, t
