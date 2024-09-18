import numpy as np

def quad(a: np.float32, b: np.float32, c: np.float32): #im enforcing the type because im a big boy
    q=np.sqrt(b**2-4*a*c)
    print('method a:')
    print( (-b+q)/ (2*a)) #correct
    print( (-b-q)/ (2*a))
    print("method b")
    print(((2*c)/(-b+q))) #correct
    print(( (2*c) /(-b-q)))
    return( (2*c) /(-b+q)),(2*c)/(-b-q)


#the problem happens when b>>4ac so you end up with something close to zero

quad(0.001,1000, 0.001)