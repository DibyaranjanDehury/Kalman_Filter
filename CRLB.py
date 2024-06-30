import numpy as np


def fcrlb(Q, noise_var):
    N = 10
    t = np.arange(0, 6635, 1)
    l = len(t)

    # Regularize the matrix Q if it is singular
    if np.linalg.cond(Q) < np.finfo(float).eps:
        Q += np.eye(Q.shape[0]) * (np.finfo(float).eps * np.max(np.abs(np.diag(Q)))) + np.eye(Q.shape[0])

    jjj = np.zeros((1, len(t)))
    for pp in range(1):
        d2 = np.zeros((2, 2, N))
        for p in range(N):
            n1 = 0 + np.sqrt(noise_var) * np.random.randn(l)  # noise signal
            n2 = 0 + np.sqrt(noise_var) * np.random.randn(l)  # noise signal
            v = np.column_stack((n1, n2))
            R = np.cov(v, rowvar=False)
            A = np.array([[1, 0], [0, 1]])
            H = np.array([[1, 0], [0, 1]])

            # CRLB
            d11 = A.T @ np.linalg.inv(Q) @ A
            d12 = -A.T @ np.linalg.inv(Q)
            d22 = np.linalg.inv(Q) + H.T @ np.linalg.inv(R) @ H
            d2[:, :, p] = d22

        d222 = np.mean(d2, axis=2)
        j = np.zeros((2, 2, len(t)))
        jin = np.zeros(len(t))
        jin[0] = 0
        for i in range(1, len(t)):
            # Regularize intermediate matrices to ensure non-singularity
            intermediate_matrix = j[:, :, i - 1] + d11
            if np.linalg.cond(intermediate_matrix) < np.finfo(float).eps:
                intermediate_matrix += np.eye(intermediate_matrix.shape[0]) * (
                            np.finfo(float).eps * np.max(np.abs(np.diag(intermediate_matrix)))) + np.eye(
                    intermediate_matrix.shape[0])

            j[:, :, i] = np.linalg.pinv(d222 - (d12 @ np.linalg.pinv(intermediate_matrix) @ d12))
            c = j[0, 0, i]
            jin[i] = c
        jjj[pp, :] = jin

    crb = jjj[0, :]
    return crb
