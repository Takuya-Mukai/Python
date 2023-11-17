def primes(x):
    e = [2]
    for y in range(3,x+1):

        s = 0
        for i in range(0,int(len(e))):
            if y % e[i] == 0:
                break
            else:
                s += 1
            if s == int(len(e)):
                e += [y]

    return e
print(primes(10000))
