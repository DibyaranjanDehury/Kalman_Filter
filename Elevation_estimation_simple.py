import numpy as np
import pandas as pd
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

# Generate noisy data
noise_var = 1
time = np.arange(len(T_r))
n1 = np.random.randn(len(T_r)) * np.sqrt(noise_var)
n2 = np.random.randn(len(T_a)) * np.sqrt(noise_var)
n3 = np.random.randn(len(T_e)) * np.sqrt(noise_var)

# Generate the measured position and velocity
m_r = T_r + n1
m_a = T_a + n2
m_e = T_e + n3

# Initialize the Kalman filter
R = np.cov(np.vstack((n1, n2, n3)))
Xe_all = np.zeros((3, len(T_r)))
z = np.vstack((m_r, m_a, m_e)).T

# Define state transition and observation matrices
A = np.eye(3)
H = np.eye(3)

# Initialize state estimate and covariance matrix
Xe = np.array([5, 355, 40])  # Initial state: [range, azimuth, elevation]
Q = 0.01 * np.eye(3)
P = np.zeros_like(Q)
Xe_all[:, 0] = Xe

# Store estimated states at each time step
for k in range(1, len(T_r)):
    if k == 0:
        x1 = A @ Xe
        P1 = A @ P @ A.T + Q
    else:
        x1 = A @ Xe_all[:, k-1]
        P1 = A @ P @ A.T + Q

    # Compute Kalman gain
    K = P1 @ H.T @ np.linalg.inv(H @ P1 @ H.T + R)

    # Update state estimate and covariance
    Xe_all[:, k] = x1 + K @ (z[k] - H @ x1)
    P = (np.eye(3) - K @ H) @ P1

MSE = np.mean((T_e - Xe_all[2, :]) ** 2)
print('Mean Squared Error for elevation :', MSE)

# Plotting 2D plots for Range, Azimuth, and Elevation
plt.figure(figsize=(12, 8))

# True Elevation w.r.t Time
plt.subplot(131)
plt.scatter(time[::8], T_e[::8], color='g', label='True Elevation', linewidth=1)
plt.xlabel('Time (Sec) - starting from 9hr 58min')
plt.ylabel('Elevation')
plt.title('True Elevation w.r.t Time')
plt.legend()

# Measured Elevation (Noisy)
plt.subplot(132)
plt.scatter(time[::8], m_e[::8], color='r', label='Measured Elevation', linewidth=1)
plt.xlabel('Time (Sec) - starting from 9hr 58min')
plt.ylabel('Elevation')
plt.title('Measured Elevation (Noisy)')
plt.legend()

# Estimated Elevation using Kalman Filtering
plt.subplot(133)
plt.scatter(time[::8], Xe_all[2, ::8], color='b', label='Estimated Elevation', linewidth=1)
plt.xlabel('Time (Sec) - starting from 9hr 58min')
plt.ylabel('Elevation')
plt.title('Estimated Elevation using Kalman Filtering')
plt.legend()

plt.tight_layout()
plt.show()
