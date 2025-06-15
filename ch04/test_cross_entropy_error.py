import numpy as np

def cross_entropy_error(y, t):
    if y.ndim == 1:
        print("t_origin = ", t)
        print("y_origin = ", y)
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        print("t_reshape = ", t)
        print("y_reshape = ", y) 

    batch_size = y.shape[0]
    print("y.shape =", y.shape)
    print("batch_size = ", batch_size)

    print(t * np.log(y + 1e-7))
    print(np.sum(t * np.log(y + 1e-7)))
    return -np.sum(t * np.log(y + 1e-7))/batch_size

y = [[0.1,0.7,0.2],
     [0.05,0.8,0.15]]
t = [[  0,  1,  0],
     [  0,  1,  0]]

print(cross_entropy_error(np.array(y),np.array(t)))