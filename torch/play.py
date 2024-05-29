x = 0

for i in range(1,10):
    num = 1000
    new_x = 0.9 * x + 0.1 * num
    print(new_x)
    print(new_x - x)
    print()
    x = new_x
    
print("final", x)