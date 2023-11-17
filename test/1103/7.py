import math
import numpy as np
import matplotlib.pyplot as plt
#def det1(e):
#    global a
#    n = e.shape[0]
#    if n == 1:
#        return e[0,0]
#    a = 0
#    for i in range(0,n):
#        a += (-1)**i*det1(np.delete(np.delete(e,0,0),i,1))
#    return a
#w = np.reshape(range(1),[1,1])
#print(det1(w))
#print(np.linalg.det(w))



def d0sin(i):
    if i % 4 == 0:
        return 0
    elif i % 4 == 1:
        return 1
    elif i % 4 == 2:
        return 0
    else:
        return -1
    
def taylor(n,x):
    y = 0
    for i in range(n):
        y += d0sin(i) * (1/math.factorial(i)) * x**i
    return y

def taylorgraph(i):
    for n in range(i):
	    x = np.linspace(0,10,100)
	    y = [taylor(n,z) for z in x]
	    plt.plot(x,y)
	    plt.show()
	

taylorgraph(10)
