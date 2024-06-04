class test:
    def __init__(self, t2, gr, **kwargs):
        print(t2, gr)
        print(kwargs)
        
class test2:
    def __init__(self):
        pass
    
t2 = test2()
test(t2, {} ,epochs=3, asdf="asdf", grag="grgr")