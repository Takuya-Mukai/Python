import numpy as np
gride = [[8,0,0,0,0,0,0,0,0],
         [0,0,3,6,0,0,0,0,0],
         [0,7,0,0,9,0,2,0,0],
         [0,5,0,0,0,7,0,0,0],
         [0,0,0,0,4,5,7,0,0],
         [0,0,0,1,0,0,0,3,0],
         [0,0,1,0,0,0,0,6,8],
         [0,0,8,5,0,0,0,1,0],
         [0,9,0,0,0,0,4,0,0]]
gride = np.array(gride)
e =np.zeros((9,9,9))
f =np.zeros((9,9))
g =np.zeros((9,9,9))


def possible(x,y,n):
    for i in range(0,9):
        if n == gride[x][i]:
            return False
    for i in range(0,9):
        if n == gride[i][y]:
            return False
    
    xr, yr = x // 3,y // 3
    for i in range(xr*3,xr*3+3):
        for j in range(yr*3,yr*3+3):
            if gride[i][j] == n:
                return False
    return True


def count():
    global e
    e.fill(0)
    for x in range(0,9):
        for y in range(0,9):
            for n in range(1,10):
                if gride[x][y] != 0:
                    e[x][y][n-1] = 2
                elif possible(x,y,n):
                    e[x][y][n-1] = 1
    return e



def sumup():
    global f
    f.fill(9)
    for x in range(0,9):
        for y in range(0,9):
            for n in range(0,9):
                if e[x][y][n] == 1:
                    f[x][y] -= 1

print(gride)

def solve():
    global e
    global f
    global g
    global gride
    

    if np.all(gride > 0):
        return gride

    count()
    sumup()
    min = np.argmin(f)
    x, y = np.unravel_index(min, f.shape)
    print('x,y =',x,',',y)

    i = 0
    for ii in range(9,0,-1):
        if g[x][y][ii-1] == 1:
            i = ii-1
            break
    print(i)

    for jj in range(i+1,9):
        if e[x][y][jj] == 1:
            gride[x][y] = jj+1
            print('number is', jj+1)
            print(gride)
            solve()
            gride[x][y] = 0
            g[x][y][jj] = 2
            print('fail')

solve()
print(e[0][1])
