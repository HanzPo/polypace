arr = []
for i in range(100):
    arr.append(-1)
    
p1=0
p2=0

for j in range(1, 10):
    print(j)
    
    cnt = 0
    while(True):
        print("p2: ", p2)
        arr[p2]=1
        p2=(1+p2)%len(arr)
        cnt+=1
        if(cnt>=j):
            break
    
        
    print(arr)

    while(True):
        print("p1: ", p1)
        arr[p1]=0
        p1=(1+p1)%len(arr)
        if(p1==p2):
            break
        
    print(arr)