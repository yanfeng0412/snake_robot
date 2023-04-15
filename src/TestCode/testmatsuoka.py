import time 
import numpy as np


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

        

    def run(self,fb = 0):
        
        self.fb1 = self.kp * fb
        self.fb2 = -self.kp * fb
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
        # return self.y1,self.y2,
        return self.theta


J = {}
j1 = Matsuoka(id = 1, tr=0.1642, ta=0.6418, beta=8.2401,  w12=5.3571, w21=5.3571, A=0.8539)
J[1] = j1.run()
print("J1 = " ,J[1])

j2 = Matsuoka(id = 2, tr=0.1276, ta=0.6971, beta=7.5937,  w12=5.4866, w21=5.4866, A=0.7666)
J[2] = j2.run()
print("J2 = " ,J[2])

j3 = Matsuoka(id = 3, tr=0.1746, ta=0.7378, beta=9.5590,  w12=5.1743, w21=5.1743, A=0.7863)
J[3] = j3.run()
print("J3 = " ,J[3])

j4 = Matsuoka(id = 4, tr=0.2367, ta=0.5084, beta=8.9907,  w12=6.0466, w21=6.0466, A=0.9515)
J[4] = j4.run()
print("J4 = " ,J[4])
j5 = Matsuoka(id = 5, tr=0.2674, ta=0.6203, beta=14.9224, w12=9.2390, w21=9.2390, A=1.5598)
J[5] = j5.run()
print("J5 = " ,J[5])

j6 = Matsuoka(id = 6, tr=0.3250, ta=0.6065, beta=16.000,  w12=9.0270, w21=9.0270, A=1.5434)
J[6] = j6.run()
print("J6 = " ,J[6])

j7 = Matsuoka(id = 7, tr=0.3527, ta=0.5620, beta=15.2292, w12=8.7594, w21=8.7594, A=1.4544)
J[7] = j7.run()
print("J7 = " ,J[7])

j8 = Matsuoka(id = 8, tr=0.2862, ta=0.7722, beta=18.2860, w12=9.8201, w21=9.8201, A=1.2749)
J[8] = j8.run()
print("J8 = " ,J[8])

if __name__ == "__main__":
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
        pos = np.zeros((9,1))
        for j in range (200):  
            t0 = time.time() 
            for i in range(1, 9):
                    # pos[i] = []
                    # pos[i].append(pos[i])
                    J[i].run(pos[i])
                    print('J',i,"=",pos[i])


           