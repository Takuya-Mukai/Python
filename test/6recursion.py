def fib(n):
    
    if n == 1 or n == 2:
        return 1
    else:
        return fib(n-1) + fib(n-2)

print(fib(5))
def exp(x):
    print(5**x)
exp(13)
def fib(n):
    fib = [0 for _ in range(n)]
    fib[0] = 1
    fib[1] = 1
    for i in range(2,n):
        fib[i] = fib[i-1] +fib[i-2]
    return fib[n-1]
print(fib(5))
    
def divs(x):
    memo = []
    for i in range(1,x+1,1):
        if x % i == 0:
            memo += [i]
    return memo

print(divs(5))

'''
for i in range(1,101):
    print(i,divs(i))

import matplotlib.pyplot as plt
x = []
y = [i for i in range(100)]
for i in range(1,101):
    x+= [len(divs(i))]
plt.bar(y,x)
plt.show()

def complete(x):
    meo = []
    for i in range(1,x+1):
        if 2*i == sum(divs(i)):
            
    return memo
print(complete(10000))
def isprime(x):
    memo = []
    for i in range(1,x):
'''
def nqueen(k):
    if (k == n ):
        print(ban)
        return
    for i in range(n):
        if(possible(k,i)):
            ban[k] = i
            nqueen(k+1)
        ban[k] = -1

def possible(k,i):
    for x in range(k):
        if ban[k] == i or ban[j] == i + 
