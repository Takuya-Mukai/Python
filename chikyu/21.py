def san(rist):
    return sum(rist)

def kika(rist):
    for i in range(len(rist)):
        rist[i] = 1/rist[i]
    return sum(rist)

rist = list(map(float,input().split()))

print(kika(rist))
print(san(rist))

