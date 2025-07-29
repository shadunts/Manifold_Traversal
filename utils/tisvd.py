import numpy as np

def TISVD(cur_x, U_old, S_old, i, d):
    '''
        This function is to be called when computing TISVD.
        This function takes one point cur_x, and efficiently computes TISVD using previous U and S matrices:

        U_old*S_old*U_old^T + cur_x * cur_x^T
        
        Inputs:
            cur_x ------ ndarray of size (D, 1), current centered point being processed
            U_old ------ orthonormal matrix of size (D, d)
            S_old ------ diagonal matrix
            i ---------- current iteration
            d ---------- intrinsic dimension
        
        Outputs:
            U_new ------------ D x d orthonormal matrix
            S_new_diag ------- a diagonal matrix of singular values, d x d
    '''

    p_x = (np.dot(U_old.T, cur_x))

    x_perp = cur_x - np.dot(U_old, p_x)
    x_perp_norm = (np.linalg.norm(x_perp))

    # expand S into an i+1 x i+1 matrix. Expan by adding a column of zeros
    S_exp_ = np.column_stack((S_old, np.zeros(S_old.shape[0])))
    S_exp = np.vstack((S_exp_, np.zeros(S_old.shape[1] + 1)))  # then add a row of zeros

    unit_norm_x_perp = (x_perp / x_perp_norm).reshape(-1, 1)
    p_x_reshaped = p_x.reshape(-1, 1)
    x_perp_norm_reshaped = np.array([x_perp_norm]).reshape(-1, 1)
    Ma = np.vstack((p_x_reshaped, x_perp_norm_reshaped))
    K = S_exp + np.dot(Ma, Ma.T)

    U_K, S_K, _ = np.linalg.svd(K)

    U_new_ = np.dot(np.hstack((U_old, unit_norm_x_perp)), U_K)
    S_new_ = S_K.copy()

    U_new = (U_new_[:, :d])
    S_new_diag = (np.diag(S_new_[:d]))

    return U_new, S_new_diag
