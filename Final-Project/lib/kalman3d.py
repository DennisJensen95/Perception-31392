import numpy as np


def update(x, P, Z, H, R, I):
    y = Z - H @ x
    S = H @ P @ np.transpose(H) + R
    K = P @ np.transpose(H) @ np.linalg.pinv(S)
    x_k = x  + K @ y
    P_k = (I - K @ H) @ P 

    return x_k, P_k


def predict(x, P, F, u):
    x_k = F @ x + u
    P_k = F @ P @ np.transpose(F)

    return x_k, P_k


def kalman(x, y, z):
    
    centroid = np.array([[x], [y], [z]])
    
    # The initial state (9x1).
    x = np.array([[0],
                  [0],
                  [0],
                  [0],
                  [0],
                  [0],
                  [0],
                  [0],
                  [0]])
    
    # The initial uncertainty (9x9).
    P = np.eye(9) * 1000
    
    # The external motion (9x1).
    u = np.array([[0],
                  [0],
                  [0],
                  [0],
                  [0],
                  [0],
                  [0],
                  [0],
                  [0]])
    
    # The transition matrix (9x9).
    delta_t = 1/2
    F = np.array([[1, 0, 0, delta_t, 0, 0, 1/2*delta_t**2, 0, 0],
                  [0, 1, 0, 0, delta_t, 0, 0, 1/2*delta_t**2, 0],
                  [0, 0, 1, 0, 0, delta_t, 0, 0, 1/2*delta_t**2],
                  [0, 0, 0, 1, 0, 0, delta_t, 0, 0],
                  [0, 0, 0, 0, 1, 0, 0, delta_t, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0, delta_t],
                  [0, 0, 0, 0, 0, 0 ,1, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 1]])
        
    # The observation matrix (3x9).
    H = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0, 0, 0, 0]])
    
    # The measurement uncertainty
    R = 1
    
    # Idendity matrix
    I = np.eye(9)
    
    
    x_pred, P_pred = predict(x, P, F, u)
    x_upd, P_upd = update(x_pred, P_pred, centroid, H, R, I)
    x, P = x_upd, P_upd
    
    center_pred = (x_pred[0], x_pred[1], x_pred[2])
    
    return center_pred
    
    



    
    
    
    
    
    
    
    
    
    
    