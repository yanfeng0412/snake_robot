
import numpy as np
TotalNumberofServos = 15
import math 
import time



def serpernoid(a,b,c,w):
    tx = np.arange(10,100,1)
    t = tx/10
    n = 16
    beta = float(b/n)
    alpha = float(2*a*abs(math.sin(beta/2)))
    gama = float(-c/n)
    theta = np.empty((n,len(t)),dtype = float)
    # q = [[0 for i in range(len(t))]for j in range(4)]
    q =  np.empty((n,len(t)), dtype = object)
    for i in range(n-1):
        for j in range(len(t)):
                x = np.dot(w, t[j])
                y = np.dot([i],beta)
                q[i][j] = 2*alpha*math.sin((x+y)+gama)
                theta[i][j] = np.asfarray(q[i][j]) 
    k = 0
    
    for j in range(len(t)):
        for k in range(8):
            
            print("theta",2*k,"=",0*180/math.pi,"theta",2*k+1,"=",theta[k,j]*180/math.pi)            
            k += 1
        print("_________________________________________________________")
        time.sleep(0.5)

serpernoid(75*math.pi/180,120*math.pi/180,0,120*math.pi/180)