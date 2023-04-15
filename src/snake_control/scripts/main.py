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
TotalNumberofServos = 8
Shift = 2*math.pi/TotalNumberofServos

jpub ={}
ns = '/snake/joint'
for i in range (TotalNumberofServos):
    jpub[i] = rospy.Publisher(ns + str(i+1)+'_position_controller/command',Float64, queue_size = 10)

def serpernoid(a,b,c,w):
    tx = np.arange(10,100,1)
    t = tx/10
    n = 7
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
        print("q",[i])
        print (q[i,:],"|")
        print ("-------------------------")
        # print("theta",[i])
        # print (theta[i,:],"|")
        # print ("-------------------------")
    
    for i in range(n-1):
        for j in range(len(t)):
            jpub[1].publish(theta[0,j])
            jpub[3].publish(theta[1,j])
            jpub[5].publish(theta[2,j])
            jpub[7].publish(theta[3,j])
            # jpub[9].publish(theta[4,j])
            # jpub[11].publish(theta[5,j])
            # ta = rospy.get_time()
            # jpub[2].publish(0.5*math.sin(0.5*ta))
            jpub[0].publish(0)
            jpub[2].publish(0)
            jpub[4].publish(0)
            jpub[6].publish(0)
            # jpub[8].publish(0)
            # jpub[10].publish(0)
            #print("theta1:",theta[0,j], "theta2:",theta[1,j],"theta3:",theta[2,j], "theta4:",theta[3,j])
            rospy.sleep(0.5)

def TranSinuous(A,w,k,gama):
    tx = np.arange(10,400,1)
    t = tx/10
    n = 8
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
        print('theta0 = ', theta[0,j]*180/math.pi, 'theta2 = ', theta[1,j]*180/math.pi,'theta4 = ', theta[2,j]*180/math.pi,'theta6 = ', theta[3,j]*180/math.pi)
        rospy.sleep(0.2)

def Sinuous(A,w,k,gama):
    ## K Change the number of wave crests 
    ## A change the Amplitude of Wave
    ## w = frequency ??
    ## gama change the direction of Motion Planing
    tx = np.arange(10,400,1)
    t = tx/10
    n = 8
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
        jpub[0].publish(0)
        jpub[2].publish(0)
        jpub[4].publish(0)
        jpub[6].publish(0)
        print('theta0 = ', theta[0,j]*180/math.pi, 'theta2 = ', theta[1,j]*180/math.pi,'theta4 = ', theta[2,j]*180/math.pi,'theta6 = ', theta[3,j]*180/math.pi)
        rospy.sleep(0.2)

def CSinuous(A,w,k,gama):
 
    tx = np.arange(10,400,1)
    t = tx/10
    n = 8
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
        
        print('theta0 = ', theta[0,j]*180/math.pi, 'theta2 = ', theta[1,j]*180/math.pi,'theta4 = ', theta[2,j]*180/math.pi,'theta6 = ', theta[3,j]*180/math.pi)
        rospy.sleep(0.2)

def StraightLine():   
    for i in range(TotalNumberofServos):
        jpub[i].publish(0)
   
def Cshape():
     for i in range(TotalNumberofServos):
         jpub[i].publish(math.pi/3)

def Ubend():
    for i in range(TotalNumberofServos):
        if (i ==4 or i ==5):
            jpub[i].publish(math.pi/2)
        else:
            jpub[i].publish(0)


def slither(offset, Amplitude, Speed, Wavelengths):
    
    theta = np.empty((TotalNumberofServos,(360)), dtype = float)
    q = np.empty((TotalNumberofServos,(360)), dtype = float)
    # while (True):
    for j in range(TotalNumberofServos):
            for i in range(0,360):
                rads = i*math.pi/180
                theta[j][i]= offset+Amplitude*math.sin(Speed*rads+j*Wavelengths*Shift)
                q[j][i] = theta[j][i]*math.pi/180 #convert to rad mode
            # time.sleep(1)
            # print("theta",[j],":",theta[j,:])
    for i in range (360):
            # print(theta[0][i],theta[1][i],theta[2][i],theta[3][i],theta[4][i],theta[5][i],theta[6][i],theta[7][i])
            print(q[0][i],q[1][i],q[2][i],q[3][i],q[4][i],q[5][i],q[6][i],q[7][i])
            jpub[0].publish(q[0][i])
            jpub[1].publish(q[1][i])
            jpub[2].publish(q[2][i])
            jpub[3].publish(q[3][i])
            jpub[4].publish(q[4][i])
            jpub[5].publish(q[5][i])
            jpub[6].publish(q[6][i])
            jpub[7].publish(q[7][i])
            rospy.sleep(0.15)
# def Trasferwave(a,b,c,w):
 

def Combination_gait(a1,b1,a2,b2):
    tx = np.arange(10,100,1)
    t = tx/10
    c = float(0)
    w = float(120*math.pi/180)
    n = 5
    beta1 = float(b1/n)
    alpha1 = float(2*a1*abs(math.sin(beta1/2)))
    gama = float(-c/n)
    theta1 = np.empty((n-1,len(t)),dtype = float)
    q1 =  np.empty((n-1,len(t)), dtype = object)
    beta2 = float(b1/n)
    alpha2 = float(2*a1*abs(math.sin(beta2/2)))
    gama = float(-c/n)
    theta2 = np.empty((n-1,len(t)),dtype = float)
    q2 =  np.empty((n-1,len(t)), dtype = object)
    for i in range(n-1):
        for j in range(len(t)):
                x1 = np.dot(w, t[j])
                y1 = np.dot([i],beta1)
                q1[i][j] = 2*alpha1*math.sin((x1+y1)+gama)
                theta1[i][j] = np.asfarray(q1[i][j]) 
                x2 = np.dot(w, t[j])
                y2 = np.dot([i],beta2)
                q2[i][j] = 2*alpha2*math.sin((x2+y2)+gama)
                theta2[i][j] = np.asfarray(q2[i][j]) 
        print("q1",[i])
        print (q1[i,:],"|")
        print ("-------------------------")
        print("q2",[i])
        print (q2[i,:],"|")
        print ("-------------------------")

    for i in range(n-1):
        for j in range(len(t)):
            jpub[1].publish(theta1[0,j])
            jpub[3].publish(theta1[1,j])
            jpub[5].publish(theta2[2,j])
            jpub[7].publish(theta2[3,j])
            jpub[2].publish(0)
            jpub[4].publish(0)
            jpub[6].publish(0)
            #print("theta1:",theta[0,j], "theta2:",theta[1,j],"theta3:",theta[2,j], "theta4:",theta[3,j])
            rospy.sleep(0.5)

        


   

if __name__ == '__main__':
    rospy.init_node('main', anonymous=True)
    rospy.sleep(1)
    rospy.loginfo("InitialRobotState")
    rospy.loginfo("ehehehehehhehehehhehehehehehehehehehe")
    try:
        # Cshape()
        # rospy.sleep(10)
        # rospy.loginfo("Cshape Finish")

        # StraightLine()
        # rospy.sleep(10)
        # rospy.loginfo("StraightLine Finish")

        # Ubend()
        # rospy.sleep(10)
        # rospy.loginfo("Ubend Finish")

        # slither(0, 80, 1, 0.5)
        # rospy.sleep(10)
        # rospy.loginfo("Curve2 motion Finish")

        # serpernoid(75*math.pi/180,120*math.pi/180,0,120*math.pi/180)
        # rospy.sleep(10)
        # rospy.loginfo("Serpernoid 1")

        # Combination_gait(75*math.pi/180,120*math.pi/180,20*math.pi/180,270*math.pi/180)

        # serpernoid(45*math.pi/180,270*math.pi/180,0,120*math.pi/180)
        # serpernoid(20*math.pi/180,270*math.pi/180,0,120*math.pi/180)
        # serpernoid(30*math.pi/180,300*math.pi/180,0,135*math.pi/180)
        # rospy.sleep(10)
        # rospy.loginfo("Serpernoid 2")

        Sinuous(1,90*math.pi/180,1.2,0)
        rospy.loginfo("Sinuous")
        rospy.sleep(5)
        
        #Trasferwave(45*math.pi/180,270*math.pi/180,0,120*math.pi/180)
        # rospy.loginfo("Transferwave")
        # rospy.sleep(5)
        
    except rospy.ROSInterruptException:
        pass