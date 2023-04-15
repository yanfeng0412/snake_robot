import rospy
import sys
import numpy as np
import time
import math
import scipy.io
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from std_msgs.msg import Float64
from fitness import fitness

jpub ={}
jsub = {}
topic_names = {}
ns = '/snake/joint'
TotalNumberofServos = 8
position = {}

for i in range (TotalNumberofServos):
    jpub[i] = rospy.Publisher(ns + str(i+1)+'_position_controller/command',Float64, queue_size = 10)
    topic_names[i] = ns+str(i+1)+'position_controller/command'
    ## Here can simplified the code as  Suffix = _position_controller/command

def callback(data,i):
    global position
    if i not in position:
        position = []
    else :
        position[i].append(data.data)

def JointSubscriber():
    for i in range(TotalNumberofServos):
        jsub[i] = rospy.Subscriber(topic_names[i],Float64, lambda data, i = i:callback(data,i))

class Network: 
    def __init__(self,n):
       # TODO: network from csv
       self.w = {}

class Matsuoka:
    def __init__(self, id, tr, ta, beta, w12, w21, A):
        weij=1
        q=1
        dt = 0.05
        s1=1
        s2=1
        kp = -0.35
        fb1 = 0
        fb2 = 0
        fk=1
        self.u1 = 0.00000001
        self.u2 = 0 
        self.v1 = 0
        self.v2 = 0

        self.y1 = 0
        self.y2 = 0

        self.id = id 
        self.fk = fk #tuning frequenct for tau_r and tau_a
        self.tau_r = tr * self.fk
        self.tau_a = ta * self.fk
        self.s1 = s1 
        self.s2 = s2
        self.fb1 = fb1 #feedback 1 
        self.fb2 = fb2 #feedback 2 

        self.beta = beta
        self.w12 = w12 
        self.w21 = w21
        self.weij = weij
        self.q = q 
        self.dt = dt
        self.A = A 

        self.kp = kp # feedback coefficient for position

        self.theta = [0]

        fb = 1 
        self.fb1 = self.kp * fb
        self.fb2 = -self.kp * fb

    def run(self):
        du1 = 1/self.tau_r * (-self.u1 - self.beta*self.v1 - self.w12*self.y2 + self.weij + self.s1 + self.fb1)
        dv1 = 1/self.tau_a * (-self.v1 + self.y1**self.q)

        du2 = 1/self.tau_r * (-self.u2 - self.beta*self.v2 - self.w21*self.y1 + self.weij + self.s2 + self.fb2)
        dv2 = 1/self.tau_a * (-self.v2 + self.y2**self.q)

        #Euler 
        self.u1 += du1*self.dt
        self.v1 += dv1*self.dt
        self.u2 += du2*self.dt
        self.v2 += dv2*self.dt

        self.y1 = max(0,self.u1)
        self.y2 = max(0,self.u2)
        self.theta.append(self.A*(self.y1-self.y2))
        # return self.theta.append(self.A*(self.y1-self.y2))
        return self.y1,self.y2,self.theta


# pos = {}
# for i in range(1, 9):
#     pos[i] = []
#     #Setting CPG oscillator output via PD controller to the joints

if __name__ == "__main__":
    rospy.init_node('main', anonymous=True)
    rospy.sleep(1)
    rospy.loginfo("InitialRobotState")
    rospy.loginfo("ehehehehehhehehehhehehehehehehehehehe")
    try:
        count = 10 
        t_init = time.time()
        while count > 0:
            count -= 1 
            J = {}
            J[1] = Matsuoka(id = 1, tr=0.1642, ta=0.6418, beta=8.2401,  w12=5.3571, w21=5.3571, A=0.8539)
            J[2] = Matsuoka(id = 2, tr=0.1276, ta=0.6971, beta=7.5937,  w12=5.4866, w21=5.4866, A=0.7666)
            J[3] = Matsuoka(id = 3, tr=0.1746, ta=0.7378, beta=9.5590,  w12=5.1743, w21=5.1743, A=0.7863)
            J[4] = Matsuoka(id = 4, tr=0.2367, ta=0.5084, beta=8.9907,  w12=6.0466, w21=6.0466, A=0.9515)
            J[5] = Matsuoka(id = 5, tr=0.2674, ta=0.6203, beta=14.9224, w12=9.2390, w21=9.2390, A=1.5598)
            J[6] = Matsuoka(id = 6, tr=0.3250, ta=0.6065, beta=16.000,  w12=9.0270, w21=9.0270, A=1.5434)
            J[7] = Matsuoka(id = 7, tr=0.3527, ta=0.5620, beta=15.2292, w12=8.7594, w21=8.7594, A=1.4544)
            J[8] = Matsuoka(id = 8, tr=0.2862, ta=0.7722, beta=18.2860, w12=9.8201, w21=9.8201, A=1.2749)
            
            for j in range(200):
                t0 = time.time()

                for i in range(1,9):
                    J[i].run(position[i][-1])

                for i in range(1,9):
                    retrunCode = jpub[i].publish(J[i])
                    # returnCode = vrep.simxSetJointTargetPosition(robot.clientID, jointsV[i], J[i].theta[-1], vrep.simx_opmode_streaming)
                t_ = time.time() - t0
                # pause .01 sec
                time.sleep(.05 - t_)
            
            print('Done',count)
        print(count,'runs cost', time.time()-t_init, 'seconds.')

        plt.plot(J[1].theta,color='r')
        plt.plot(position[1],color='b')

        plt.show()
    except rospy.ROSInterruptException:
        pass

        

