def count_bits(k):
    binary = bin(k)
    strin = [ones for ones in binary[2:] if ones == '1']
    print(len(strin))


t = int(input())
for i in range(t):
    n = int(input())
    count_bits(n)