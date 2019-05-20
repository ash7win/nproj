from itertools import permutations


def special_count(f, k):
    l = []
    for i in range(f):
        l.append(i)
    perm = permutations(l)
    for j in list(perm):
        count = 0
        s = 0
        for e in range(1, f):
            a = abs(j[e - 1] - e)
            count += a
        if count == k:
            s += 1
            break
        else:
            continue
    if s == 0:
        print('NO')
    else:
        print('YES')


t = input()
for i in range(int(t)):
    n, k = map(int, input().split())
    special_count(n, k)