import numpy as np

def P3P(Pc, Pw, K=np.eye(3)):
    """
    Solve Perspective-3-Point problem, given correspondence and intrinsic

    Input:
        Pc: 4x2 numpy array of pixel coordinate of the Bril tag corners in (x,y) format
        Pw: 4x3 numpy array of world coordinate of the Bril tag corners in (x,y,z) format
    Returns:
        R: 3x3 numpy array describing camera orientation in the world (R_wc)
        t: (3,) numpy array describing camera translation in the world (t_wc)

    """

    ##### STUDENT CODE START #####
    P = Pw[:3]
    P0 = P[0]
    P1 = P[1]
    P2 = P[2]

    a = np.linalg.norm(P1 - P2)
    b = np.linalg.norm(P0 - P2)
    c = np.linalg.norm(P0 - P1)

    Pc = Pc - K[:2, 2].T    
    f = (K[0, 0] + K[1, 1]) / 2
    uv = np.hstack((Pc[:3], np.ones((3, 1)) * f))
    j = uv / np.linalg.norm(uv, axis = 1, keepdims=True)

    CA = np.dot(j[1], j[2])
    CB = np.dot(j[0], j[2])
    CG = np.dot(j[0], j[1])

    A = (a ** 2 - c ** 2) / b ** 2
    B = (a ** 2 + c ** 2) / b ** 2
    
    A4 = (A - 1) ** 2 - 4 * ( c ** 2 / b ** 2) * CA ** 2
    A3 = 4 * (A * (1 - A) * CB - (1 - B) * CA * CG + 2 * (c ** 2 / b ** 2) * CA ** 2 * CB)
    A2 = 2 * (A ** 2 - 1 + 2 * A ** 2 * CB ** 2 + 2 * ((b ** 2 - c ** 2) / b ** 2) * CA ** 2 - 4 * B * CA * CB * CG + 2 * ((b ** 2 - a ** 2) / b ** 2) * CG ** 2)
    A1 = 4 * (-A * (1 + A) * CB + 2 * (a ** 2 / b ** 2) * CG ** 2 * CB - (1 - B) * CA * CG)
    A0 = (1 + A) ** 2 - 4 * (a ** 2 / b ** 2) * CG ** 2

    v = np.roots(np.array([A4, A3, A2, A1, A0]))
    v = np.real(v[np.isreal(v)])
    u = ((-1 + A) * v ** 2 - 2 * A * CB * v + 1 + A) / (2 * (CG - v * CA))

    s = np.zeros((3, 2))
    s[0] = np.sqrt(c ** 2 / (1 + u ** 2 - 2 * u * CG))
    s[1] = u * s[0]
    s[2] = v * s[0]

    s = s[:, 0] if np.all(s[:,0] > 0) else s[:, 1]
    P_c = s.reshape(3,1) * j
    R, t = Procrustes(P_c, Pw[:3, :]) 

    return R, t

def Procrustes(X, Y):
    """
    Solve Procrustes: Y = RX + t

    Input:
        X: Nx3 numpy array of N points in camera coordinate (returned by your P3P)
        Y: Nx3 numpy array of N points in world coordinate
    Returns:
        R: 3x3 numpy array describing camera orientation in the world (R_wc)
        t: (3,) numpy array describing camera translation in the world (t_wc)

    """
    a21 = Y[1,:] - Y[0,:]
    a31 = Y[2,:] - Y[0,:]
    a = np.cross(a21, a31)
    A = np.vstack([a21, np.cross(a, a21), a]).T
    
    b21 = X[1,:] - X[0,:]
    b31 = X[2,:] - X[0,:]
    b = np.cross(b21, b31)
    B = np.vstack([b21, np.cross(b, b21), b]).T
    
    M = np.matmul(B, A.T)
    U, S, VT = np.linalg.svd(M)
    V = VT.T
    
    d = np.eye(3)
    d[-1, -1] = np.linalg.det(np.matmul(V, U.T))
    R = np.matmul(np.matmul(V, d), U.T)
    t = Y.mean(axis=0) - R.dot(X.mean(axis=0)) 
    ##### STUDENT CODE END #####
    
    return R, t