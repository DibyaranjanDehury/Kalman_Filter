import numpy as np
import pandas as pd
from CRLB import fcrlb
import matplotlib.pyplot as plt

# Read the Excel file
filename = 'data.xlsx'
sheet = 0
columns_range = 'A:E'
data = pd.read_excel(filename, sheet_name=sheet, usecols=columns_range, engine='openpyxl')

# Extracting every other row from the dataframe
data = data[1::2]

# True Range, Azimuth, and Elevation
T_r = data.iloc[:, 0]
T_a = data.iloc[:, 1]
T_e = data.iloc[:, 2]

# Check for NaN values in true range, azimuth, and elevation arrays
if T_r.isnull().values.any() or T_a.isnull().values.any() or T_e.isnull().values.any():
    print("There are NaN values in the true range, azimuth, or elevation arrays!")
    # Handle NaN values (e.g., remove rows with NaN values)
    data.dropna(inplace=True)
    T_r = data.iloc[:, 0]
    T_a = data.iloc[:, 1]
    T_e = data.iloc[:, 2]

T = 0.5
t = np.arange(0, 6635)
noise_var = 1
n1 = np.random.randn(len(T_r)) * np.sqrt(noise_var)
n2 = np.random.randn(len(T_a)) * np.sqrt(noise_var)
n3 = np.random.randn(len(T_e)) * np.sqrt(noise_var)
v = np.vstack((n1, n2)).T  # Transpose to align dimensions for covariance calculation
R = np.cov(v, rowvar=False)
l = len(t)
N = 4  # window_size
it, c1, c2 = 0, 0, 0
m = np.zeros((2, 5))
flag = 0

for s in range(0, 5):
    # Generate the measured position and velocity
    m_r = T_r + n1
    m_a = T_a + n2
    m_e = T_e + n3
    innovation = np.zeros((2, len(t)))

    # Initialize the Kalman filter
    Xe_all = np.zeros((2, len(T_r)))
    z = np.vstack((m_r, m_a)).T

    # Define state transition and observation matrices
    A = np.eye(2)
    H = np.eye(2)

    # Initialize state estimate and covariance matrix
    Xe = np.array([5.04, 355])  # Initial state: [range, azimuth, elevation]
    Q = 1 * np.eye(2)
    P = np.zeros_like(Q)
    Xe_all[:, 0] = Xe

    # Store estimated states at each time step
    for k in range(0, (len(T_r))):
        if k == 0:
            x1 = A @ Xe
            P1 = A @ P @ A.T + Q
        else:
            x1 = A @ Xe_all[:, k - 1]
            P1 = A @ P @ A.T + Q

        # Compute Kalman gain
        K = P1 @ H.T @ np.linalg.inv(H @ P1 @ H.T + R)

        # Update state estimate and covariance
        Xe_all[:, k] = x1 + K @ (z[k] - H @ x1)
        P = (np.eye(2) - K @ H) @ P1
        innovation[:, k] = z[k] - H @ x1

        if k > N:
            c_est = np.zeros((2, 2))
            j0 = k - N + 1
            for j in range(j0, k + 1):
                c_est += np.outer(innovation[:, j], innovation[:, j])
            c_est /= N
            Q = K @ c_est @ K.T
    flag = flag + 1
    error = Xe_all - np.vstack((T_r, T_a))
    MSE = (error @ error.T) / 400
    m[0, s] = MSE[0, 0]
    m[1, s] = np.sum(-Xe_all[0, :] + T_r) / l
    cb = fcrlb(Q, noise_var)
    if cb[100] < m[0, s] - m[1, s]:
        if s == 0:
            N = 1
        elif m[0, s - 1] > m[0, s]:
            N += 4
            c1 += 1
        else:
            N -= 2
            c2 += 1
    else:
        break

    it += 1


print(MSE)

plt.figure(figsize=(12, 8))

# True Azimuth w.r.t Time
plt.subplot(131)
plt.scatter(t[::8], T_a[::8], color='g', label='True Azimuth', linewidth=1)
plt.xlabel('Time (Sec) - starting from 9hr 58min')
plt.ylabel('Azimuth')
plt.title('True Azimuth w.r.t Time')
plt.legend()

# Measured Azimuth (Noisy)
plt.subplot(132)
plt.scatter(t[::8], m_a[::8], color='r', label='Measured Azimuth', linewidth=1)
plt.xlabel('Time (Sec) - starting from 9hr 58min')
plt.ylabel('Azimuth')
plt.title('Noisy Azimuth w.r.t Time')
plt.legend()

# Estimated Azimuth using Kalman Filtering
plt.subplot(133)
plt.scatter(t[::8], Xe_all[1, ::8], color='b', label='Estimated Azimuth', linewidth=1)
plt.xlabel('Time (Sec) - starting from 9hr 58min')
plt.ylabel('Azimuth')
plt.title('Estimated Azimuth using Kalman Filtering')
plt.legend()

plt.figure(figsize=(12, 8))

# True Range w.r.t Time
plt.subplot(131)
plt.scatter(t[::8], T_r[::8], color='g', label='True Range', linewidth=1)
plt.xlabel('Time (Sec) - starting from 9hr 58min')
plt.ylabel('Azimuth')
plt.title('True Range w.r.t Time')
plt.legend()

# Measured Range (Noisy)
plt.subplot(132)
plt.scatter(t[::8], m_r[::8], color='r', label='Measured Range', linewidth=1)
plt.xlabel('Time (Sec) - starting from 9hr 58min')
plt.ylabel('Range')
plt.title('Noisy Range w.r.t Time')
plt.legend()

# Estimated Range using Kalman Filtering
plt.subplot(133)
plt.scatter(t[::8], Xe_all[0, ::8], color='b', label='Estimated Range', linewidth=1)
plt.xlabel('Time (Sec) - starting from 9hr 58min')
plt.ylabel('Range')
plt.title('Estimated Range using Kalman Filtering')
plt.legend()

plt.tight_layout()
plt.show()


