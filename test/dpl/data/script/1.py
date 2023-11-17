a = 'datascience'
print(len(a))
c = list(map(lambda x : print(x), a))
print(sum(list(i for i in range(51))))
def prime(n):
    memo = [True]*(n+1)
    memo[0] = False
    memo[1] = False
    for i in range(2,int(n**0.5)+1):
        if memo[i]:
            for j in range(i**2,n+1,i):
                memo[j] = False
    for i in range(n+1):
        if memo[i]:
            print(i)
prime(101)

