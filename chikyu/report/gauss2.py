import numpy as np


a = np.array([[10, 1, 4, 0],
              [1, 10, 5, -1],
              [4, 5, 10, 7],
              [0, -1, 7, 9]]
             )
c = np.array([15, 15, 26, 15]
             )
b = np.zeros((4, 4))
# a = np.array([[1, 1,
# a = np.array([[1, 1, 1],
#               [2, 2, 1],
#               [2, 3, 2]])
# b = np.zeros((3, 3))
# c = np.array([0, 3, 1])


# b  is for remembering the max value of
def scaling(a, b, c):
    # a is matrix
    n = a.shape[0]
    for i in range(n):
        # list for remembering max value
        l = np.abs(a[i])

        maxnum = a[i][0]
        for j in range(n):
            if maxnum < a[i][j]:
                maxnum = a[i][j]

        a[i] = a[i] / maxnum
        c[i] = c[i] / maxnum
    return a, b, c


def pivoting(a, b, c, x):
    # extract the max value of column from x
    max = a[x:, x]
    k = np.argmax(max)
    a[(x, k+x), :] = a[(k+x, x), :]
    c[x], c[k+x] = c[k+x], c[x]
    # print(a)
    return a, b, c


def delete(a, b, c, x):
    n = a.shape[0]
    if a[x, x] == 0 or x+1 == n:
        return
    for i in range(x+1, n):
        if a[x, x] != 0 and a[i, x] != 0:
            b[i, x] = -a[i, x]/a[x, x]
            a[i, :] = a[i]-(a[i, x]/a[x, x])*a[x]
            c[i] = c[i]+b[i, x]*c[x]
            print(c)
        else:
            pass
    return a, b, c


def gaussdelete64(a, b, c):
    a = np.array(a, dtype="float64")
    b = np.array(b, dtype="float64")
    c = np.array(c, dtype="float64")
    n = a.shape[0]
    scaling(a, b, c)
    for i in range(n-1):
        p = pivoting(a, b, c, i)
        a, b, c = p[0], p[1], p[2]
        d = delete(a, b, c, i)
        a, b, c = d[0], d[1], d[2]
    return a, b, c


def gaussdelete32(a, b, c):
    a = np.array(a, dtype="float32")
    b = np.array(b, dtype="float32")
    c = np.array(c, dtype="float32")
    n = a.shape[0]
    print(a)
    scaling(a, b, c)
    for i in range(n-1):
        p = pivoting(a, b, c, i)
        a, b, c = p[0], p[1], p[2]
        d = delete(a, b, c, i)
        a, b, c = d[0], d[1], d[2]
    return a, b, c


def gauss32(a, b, c):
    g = gaussdelete32(a, b, c)
    a = g[0]
    b = g[1]
    c = g[2]
    print(a, b, c)
    n = a.shape[0]
    x = np.zeros(n, dtype='float32')
    ans = 0
    for i in range(n-1, -1, -1):
        ans = (c[i] - sum(a[i, j] * x[j] for j in range(i, n))) / a[i, i]
        x[i] = ans
        ans = 0
    return x


def gauss64(a, b, c):
    g = gaussdelete64(a, b, c)
    a = g[0]
    b = g[1]
    c = g[2]
    print(a, b, c)
    n = a.shape[0]
    x = np.zeros(n, dtype='float64')
    ans = 0
    for i in range(n-1, -1, -1):
        ans = (c[i] - sum(a[i, j] * x[j] for j in range(i, n))) / a[i, i]
        x[i] = ans
        ans = 0
    return x

print(gauss32(a, b, c))
print(gauss64(a, b, c))
