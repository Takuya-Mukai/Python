r = float(input('r='))
eps = 0.000005
x = r
i = 0
while abs(x **2-r)>=eps:
    x = (x + r / x) / 2
    i += 1
print(x,r **0.5)
print(abs(r ** 0.5 -x))


