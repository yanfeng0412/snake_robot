# import numpy as np

# class Matsuoka:
#     def __init__(self, w1, w2, z1, z2, b, g):
#         self.w1 = w1
#         self.w2 = w2
#         self.z1 = z1
#         self.z2 = z2
#         self.b = b
#         self.g = g
        
#     def step(self, x):
#         dxdt = -self.w1 * self.z1 - self.b * self.z2 + x
#         dydt = -self.w2 * self.z2 - self.b * self.z1 + self.g * self.z1
        
#         self.z1 += dxdt * dt
#         self.z2 += dydt * dt
        
#         return self.z1 - self.z2

# # Define simulation parameters
# dt = 0.01 # time step
# T = 10 # simulation time
# N = int(T / dt) # number of time steps

# # Define Matsuoka neuron parameters
# w1 = 10.0
# w2 = 10.0
# z1 = 0.0
# z2 = 0.0
# b = 1.0
# g = 1.0

# ## g  ==> positive feedback term 
# ## x inpput signal to neuron 
# ## w1 w2 the strengths of the inhibitory connections betweent the two neuron

# # Create Matsuoka neuron object
# mn = Matsuoka(w1, w2, z1, z2, b, g)

# # Define input signal (for example, a sine wave)
# t = np.linspace(0, T, N)
# x = np.sin(2 * np.pi * 2.5 * t)

# # Simulate Matsuoka neuron and output signal
# y = np.zeros_like(x)
# for i in range(N):
#     y[i] = mn.step(x[i])
    
# # Plot input and output signals
# import matplotlib.pyplot as plt
# plt.plot(t, x, label='Input')
# plt.plot(t, y, label='Output')
# plt.legend()
# plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Parameters
a = 1.0
b = 2.5
tau = 1.0
w = 1.0
z = 0.0

# Time range
t_start = 0.0
t_end = 10.0
dt = 0.01
t = np.arange(t_start, t_end, dt)

# Simulate differential equations
y1 = np.zeros_like(t)
y2 = np.zeros_like(t)
y1[0] = 0  # initial conditions
y2[0] = 0
for i in range(1, len(t)):
    dy1dt = -y1[i-1] - a*z + b*y2[i-1] + w
    dy2dt = -y2[i-1] - a*z + b*y1[i-1]
    y1[i] = y1[i-1] + dy1dt*dt
    y2[i] = y2[i-1] + dy2dt*dt
    z = max(0, y1[i])  # nonlinear function

# Plot result
plt.plot(t, y1, label='y1')
plt.plot(t, y2, label='y2')
plt.xlabel('Time')
plt.ylabel('Output')
plt.legend()
plt.show()

