def fibmemo(n):
    memo = dict()
    memo[0] = 1
    memo[1] = 1
    if n in memo:
        return memo[n]
    else:
        memo[n] = fibmemo(n-1) + fibmemo(n-2)
    return memo[n]
print(fibmemo(20))

