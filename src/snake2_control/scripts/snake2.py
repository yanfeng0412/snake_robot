#!/usr/bin/env python
# license removed for brevity
import rospy
import math
import numpy as np
import array as arr
from std_msgs.msg import Float64
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
import ros
import time
TotalNumberofServos = 16
Shift = 2*math.pi/TotalNumberofServos

jpub ={}
ns = '/snake2/joint'
for i in range (TotalNumberofServos):
    jpub[i] = rospy.Publisher(ns + str(i+1)+'_position_controller/command',Float64, queue_size = 10)

def serpernoid(a,b,c,w):
    tx = np.arange(10,100,1)
    t = tx/10
    n = 10
    beta = float(b/n)
    alpha = float(2*a*abs(math.sin(beta/2)))
    gama = float(-c/n)
    theta = np.empty((n-1,len(t)),dtype = float)
    # q = [[0 for i in range(len(t))]for j in range(4)]
    q =  np.empty((n-1,len(t)), dtype = object)
    for i in range(n-1):
        for j in range(len(t)):
                x = np.dot(w, t[j])
                y = np.dot([i],beta)
                q[i][j] = 2*alpha*math.sin((x+y)+gama)
                theta[i][j] = np.asfarray(q[i][j]) 
        # print("q",[i])
        # print (q[i,:],"|")
        # print ("-------------------------")
        # print("theta",[i])
        # print (theta[i,:],"|")
        # print ("-------------------------")
    # k = 0
    # for k in range(8):
    for j in range(len(t)):
            # print("theta",k,"=",theta[k,j]*180/math.pi)
            # print("theta",2*k+1,"=",theta[k,j]*180/math.pi)
            jpub[1].publish(theta[1,j])
            jpub[3].publish(theta[2,j])
            jpub[5].publish(theta[3,j])
            jpub[7].publish(theta[4,j])
            jpub[9].publish(theta[5,j])
            jpub[11].publish(theta[6,j])
            jpub[13].publish(theta[7,j])
            jpub[15].publish(theta[8,j])
            
            # jpub[2*k+1].publish(theta[k,j])
            # print("theta",2*k,"=",0*180/math.pi)
            # jpub[2*k].publish(0)
            # k += 1
            rospy.sleep(0.5)


def TranSinuous(A,w,k,gama):
    tx = np.arange(10,400,1)
    t = tx/10
    n = 10
    theta = np.empty((n,len(t)),dtype = float)
    q =  np.empty((n,len(t)), dtype = object)
    for i in range(n):
        for j in range(len(t)):
                alpha = np.dot(w,t[j])
                beta = np.dot(k,[i])
                q[i][j] = A*math.sin(alpha+beta)+gama
                theta[i][j] = np.asfarray(q[i][j]) 

    # for i in range(n-1):
    for j in range(len(t)):
        jpub[0].publish(theta[0,j])
        jpub[2].publish(theta[1,j])
        jpub[4].publish(theta[2,j])
        jpub[6].publish(theta[3,j])
        jpub[8].publish(theta[4,j])
        jpub[10].publish(theta[5,j])
        jpub[12].publish(theta[6,j])
        jpub[14].publish(theta[7,j])
        
        print('theta0 = ', theta[0,j]*180/math.pi, 'theta2 = ', theta[1,j]*180/math.pi,'theta4 = ', theta[2,j]*180/math.pi,'theta6 = ', theta[3,j]*180/math.pi)
        print('theta8 = ', theta[4,j]*180/math.pi, 'theta10 = ', theta[5,j]*180/math.pi,'theta12 = ', theta[6,j]*180/math.pi,'theta14 = ', theta[7,j]*180/math.pi)
        rospy.sleep(0.2)

def Sinuous(A,w,k,gama):
    ## K Change the number of wave crests 
    ## A change the Amplitude of Wave
    ## w = frequency ??
    ## gama change the direction of Motion Planing
    tx = np.arange(10,400,1)
    t = tx/10
    n = 10
    theta = np.empty((n,len(t)),dtype = float)
    q =  np.empty((n,len(t)), dtype = object)
    for i in range(n):
        for j in range(len(t)):
                alpha = np.dot(w,t[j])
                beta = np.dot(k,[i])
                q[i][j] = A*math.sin(alpha+beta)+gama
                theta[i][j] = np.asfarray(q[i][j]) 

    # for i in range(n-1):
    for j in range(len(t)):
        jpub[1].publish(theta[0,j])
        jpub[3].publish(theta[1,j])
        jpub[5].publish(theta[2,j])
        jpub[7].publish(theta[3,j])
        jpub[9].publish(theta[4,j])
        jpub[11].publish(theta[5,j])
        jpub[13].publish(theta[6,j])
        jpub[15].publish(theta[7,j])
        jpub[0].publish(math.pi/8)
        jpub[2].publish(0)
        jpub[4].publish(0)
        jpub[6].publish(0)
        jpub[8].publish(0)
        jpub[10].publish(0)
        jpub[12].publish(0)
        jpub[14].publish(0)
        
        print('theta0 = ', theta[0,j]*180/math.pi, 'theta2 = ', theta[1,j]*180/math.pi,'theta4 = ', theta[2,j]*180/math.pi,'theta6 = ', theta[3,j]*180/math.pi)
        rospy.sleep(0.2)

def CSinuous(A,w,k,gama):
 
    tx = np.arange(10,400,1)
    t = tx/10
    n = 16
    theta = np.empty((n,len(t)),dtype = float)
    q =  np.empty((n,len(t)), dtype = object)
    for i in range(n):
        for j in range(len(t)):
                alpha = np.dot(w,t[j])
                beta = np.dot(k,[i])
                q[i][j] = A*math.sin(alpha+beta)+gama
                theta[i][j] = np.asfarray(q[i][j]) 

    # for i in range(n-1):
    for j in range(len(t)):
        jpub[0].publish(theta[0,j])
        jpub[1].publish(theta[1,j])
        jpub[2].publish(theta[2,j])
        jpub[3].publish(theta[3,j])
        jpub[4].publish(theta[4,j])
        jpub[5].publish(theta[5,j])
        jpub[6].publish(theta[6,j])
        jpub[7].publish(theta[7,j])
        jpub[8].publish(theta[8,j])
        jpub[9].publish(theta[9,j])
        jpub[10].publish(theta[10,j])
        jpub[11].publish(theta[11,j])
        jpub[12].publish(theta[12,j])
        jpub[13].publish(theta[13,j])
        jpub[14].publish(theta[14,j])
        jpub[15].publish(theta[15,j])

        
        print('theta0 = ', theta[0,j]*180/math.pi, 'theta2 = ', theta[1,j]*180/math.pi,'theta4 = ', theta[2,j]*180/math.pi,'theta6 = ', theta[3,j]*180/math.pi)
        rospy.sleep(0.2)
 

        
   

if __name__ == '__main__':
    rospy.init_node('main', anonymous=True)
    rospy.sleep(1)
    rospy.loginfo("InitialRobotState")
    rospy.loginfo("ehehehehehhehehehhehehehehehehehehehe")
    try:
        # serpernoid(75*math.pi/180,120*math.pi/180,0,120*math.pi/180)
        # rospy.sleep(10)
        # rospy.loginfo("Serpernoid 1")

        # Combination_gait(75*math.pi/180,120*math.pi/180,20*math.pi/180,270*math.pi/180)

        # serpernoid(45*math.pi/180,360*math.pi/180,0,120*math.pi/180)
        # serpernoid(45*math.pi/180,270*math.pi/180,0,120*math.pi/180)
        # serpernoid(20*math.pi/180,270*math.pi/180,0,120*math.pi/180)
        # serpernoid(30*math.pi/180,300*math.pi/180,0,135*math.pi/180)
        # rospy.sleep(10)
        # rospy.loginfo("Serpernoid 2")

        TranSinuous(0.5,100*math.pi/180,1.3,0)
        # Sinuous(1.0,100*math.pi/180,0.8,0)
        # Sinuous(0.8,100*math.pi/180,0.8,0)
        # Sinuous(1.0,100*math.pi/180,1,0)
        # rospy.loginfo("Sinuous")
        # rospy.sleep(5)
        
        #Trasferwave(45*math.pi/180,270*math.pi/180,0,120*math.pi/180)
        # rospy.loginfo("Transferwave")
        # rospy.sleep(5)
        
    except rospy.ROSInterruptException:
        pass