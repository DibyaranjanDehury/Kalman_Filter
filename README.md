# Kalman_Filter
This project is an implementation of the Kalman Filter to optimize the noisy data received from the sensors and to minimize the errors between the actual patha and the measured path.
The position of a UAV Drone can be measured by three variables(range, position, azimuth),these varaiables are measured by sensors and then the Kalman filter code minimizes the errors in the measurements.
## WHAT IS A KALMAN FILTER
The Kalman filter, named after Rudolf E. Kálmán, is an efficient recursive algorithm used to estimate the state of a dynamic system from a series of noisy measurements. It operates in two phases: prediction and update. During the prediction phase, it estimates the current state and its uncertainty. In the update phase, it incorporates new measurements to refine the state estimate, reducing uncertainty.
## BLOCK DIAGRAM
![image](https://github.com/DibyaranjanDehury/Kalman_Filter/assets/174172160/a25b2954-ec36-4e95-be6e-5c9abc7cba97)
## MATHEMATICAL MODEL
![image](https://github.com/DibyaranjanDehury/Kalman_Filter/assets/174172160/222b3a7b-3b7a-4439-a9ec-d1caf65b7a09)
## IMPLEMENTATION OF CRLB IN KALMAN FILTER
The Cramer-Rao Lower Bound (CRLB) provides a lower bound on the variance of unbiased estimators. Extending the Kalman filter with CRLB involves evaluating the theoretical limits of the estimation accuracy provided by the Kalman filter.

The Kalman filter produces estimates of the state of a linear dynamic system, and the accuracy of these estimates can be quantified by the estimation error covariance matrix. The CRLB offers a theoretical minimum for this estimation error covariance matrix. For linear systems, the CRLB is particularly relevant as it indicates the best achievable performance of any unbiased estimator, given the data.
​
