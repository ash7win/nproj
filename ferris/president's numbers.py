n, b = input().split()

for i in range(int(n),int(b)):
    flag = 0
    for j in range(2,i):
        if i % j==0:
            flag = 1
    if flag==0:
        su = 0
        flog= True
        for f in str(i):
            su+=int(f)
        for bum in range(2,su):
            flog = True
            if su % bum ==0:
                flog = False
            if su==2 or su==3:
                flog=True
            if su%2==0 or su% 3==0:
                flog=False
        if flog is True:
            print(i,end = " ")