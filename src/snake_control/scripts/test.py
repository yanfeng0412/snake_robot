import numpy as np
import time
import math
# def __init__(self, **kwargs):
#         self.parameters = {}

#         self.parameters['swing_scale'] = 1
#         self.parameters['vx_scale'] = 0.5
#         self.parameters['vy_scale'] = 0.5
#         self.parameters['vt_scale'] = 0.4

#         for k, v in kwargs.items():
#             self.parameters[k] = v
#             print("parameters:"%(k,kwargs[k]))

# rows = 3
# cols = 4
# arr = [[0 for i in range(cols)]for j in range(rows)]

# for i in range(rows):
#     for j in range(cols):
#         arr[i][j] = i*j
#         print(arr)


# tx = np.arange(0,100,1)
# t = tx/10
# a = float(75*math.pi/180)
# b = float(120*math.pi/180)
# c = float(0) 
# w = float(120*math.pi/180)
# n = 5
# beta = float(b/n)
# alpha = float(2*a*abs(math.sin(beta/2)))
# gama = float(-c/n)
# # q = [[0 for i in range(len(t))]for j in range(4)]
# q =  np.empty((n-1,len(t)), dtype = object)
# theta = np.empty((n-1,len(t)), dtype = float)

# for i in range(n-1):
#     for j in range(len(t)):
#             x = np.dot(w, t[j])
#             y = np.dot([i],beta)
#             q[i][j] = 2*alpha*math.sin((x+y)+gama)
#             theta[i][j] = np.asfarray(q[i][j])
#     print("q",[i])
#     print (q[i,:],"|")
#     print ("-------------------------")
#     print("theta",[i])
#     print (theta[i,:],"|")
#     print ("-------------------------")


# for i in range(n-1):
#     print("q",[i])
#     print (q[i,:],"|")
#     print ("-------------------------")
# python 2D array 
# import numpy as np 
# npose = 4
# nsmile = 10
# pose_cell = np.empty((npose,nsmile),dtype = object) #Create 5rows 2 colmuns empty array, type =  object  
# for i in range(npose):
#     for k in range(nsmile):
#         pose_cell[i][k] = i+1
#         print (pose_cell[0,:],"|")
#         print("---------------------------------")
#         print (pose_cell[1,:])
#         print("---------------------------------")
#         print (pose_cell[2,:])
#         print("---------------------------------")
#         print (pose_cell[3,:])
#         print("---------------------------------")
        
# print (pose_cell.shape)

# import numpy as np 
# t1 = np.arange(1,10,0.1)
# print(t1)

# t2 = np.arange(10,100,1)
# tx =t2/10
# print(tx)



# slither test!!!!!!!!!!!!!!!
# def slither(offset, Amplitude, Speed, Wavelengths):
#     TotalNumberofServos = 8
#     Shift = 2*math.pi/TotalNumberofServos
#     theta = np.empty((TotalNumberofServos,(360)), dtype = float)
#     q = np.empty((TotalNumberofServos,(360)), dtype = float)
#     for j in range(TotalNumberofServos):
#         for i in range(0,360):
#             rads = i*math.pi/180
#             theta[j][i]= -(offset+Amplitude*math.sin(Speed*rads+j*Wavelengths*Shift))
#             q[j][i] = theta[j][i]#*math.pi/180 #convert to rad mode
#         # time.sleep(1)
#         # print("theta",[j],":",theta[j,:])
#     for i in range (360):
#         # print(theta[0][i],theta[1][i],theta[2][i],theta[3][i],theta[4][i],theta[5][i],theta[6][i],theta[7][i])
#         print(q[0][i],q[1][i],q[2][i],q[3][i],q[4][i],q[5][i],q[6][i],q[7][i])
#         time.sleep(0.1)

# slither(10, 35, 2, 1.5)



# theta = 0 
# while (True):
#     while(theta < 90):
#         theta += 1 
#         print(theta)
#         time.sleep(0.1)
#     while(theta > 0):
#         theta -= 1
#         print(theta)
#         time.sleep(0.1)


# TotalNumberofServos = 9
# # for i in range(TotalNumberofServos):
# #     for j in range (360):
# #         theta1 = math.sin(60*j)
# #         theta2 = math.cos(60*j)
# #         print(theta1, theta2)
# #         print(theta1*180/math.pi,theta2*180/math.pi)

# def Sinuous(A,w,k,gama):
#     ## K Change the number of wave crests 
#     ## A change the Amplitude of Wave
#     ## w = frequency ??
#     ## gama change the direction of Motion Planing
#     tx = np.arange(10,100,1)
#     t = tx/10
#     n = 8
#     theta = np.empty((n,len(t)),dtype = float)
#     q =  np.empty((n,len(t)), dtype = object)
#     for i in range(n):
#         for j in range(len(t)):
#                 alpha = np.dot(t[j],w)
#                 beta = np.dot(k,[i])
#                 q[i][j] = A*math.sin(alpha+beta)+gama
#                 # q[i][j] = A*math.sin(w*t[j]+k*[i])+gama
#                 theta[i][j] = np.asfarray(q[i][j]) 
#     for i in range(n-1):
#         # for j in range(len(t)):
#         # print('theta0 = ', theta[0,j]*180/math.pi, 'theta2 = ', theta[1,j]*180/math.pi,'theta4 = ', theta[2,j]*180/math.pi,'theta6 = ', theta[3,j]*180/math.pi)
#       print('theta',i,"=" ,theta[i,:]*180/math.pi)

#     rows = len(theta)
#     cols = len(theta[0])
#     print('ROWS = ', rows)
#     print('Columns = ' , cols)
#     print ('Legth t =', len(t))


# Sinuous(0.5,30*math.pi/180,1.2,0)

global positions 
position = [0]*20
print(position)