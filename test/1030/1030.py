m,n = map(int, input('m, n =').split(','))

ms = m
ns = n

while True:
    k = m % n
    if k == 0:
        break
    m = n
    n = k

print(ms, 'and', ns ,'of greatest common devisor is', n)

