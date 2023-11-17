import numpy as np
# dict for remember max number
mnum = {}

def scaling(a):
  # a is matrix, x is where to start
  global mnum
  n = a.shape[0]
  #list for remember max value
  for i in range(n):
    l = np.abs(a[i])
    maxnum = max(l)
    a[i] = a[i] / maxnum
    mnum['No{}'.format(i)] = maxnum

  #scaling and memo

def pivoting(a,x):
  #remember max value
  n = a.shape[0]
  max = a[:,x]
  k = np.argmax(max)
  #list for swap
  a[[x, k], :] = a[[k, x], :]



def delete(a,x):
  n = a.shape[0]
  global b
  if a[x,x] == 0 or x+1 == n:
    return
  for i in range(x+1,n):
    if a[x,x] != 0 and a[i,x] != 0:
      b[i, x] = -a[i, x]/a[x, x]
      a[i] = a[i]-(a[i, x]/a[x, x])*a[x]
    else:
      pass




def gaussdelete(a):
  a = np.array(a,dtype = float)
  n = a.shape[0]
  scaling(a)
  print(a)
  for i in range(n-1):
    pivoting(a,i)
    print(a)
    print()
    delete(a,i)
    print(a)
    print()
  return a

a = [[10,1,4,0],[1,10,5,-1],[4,5,10,7],[0,-1,7,9]]
c = [[4,3,2],[5,10,1],[3,2,6]]
b = np.zeros((4,4))
m = np.zeros(4)
print(gaussdelete(a))
print(mnum)
print(b)

def gauss(a):
  gaussdelete(a)
  x = []
  for i in range(a.shape[0]):
