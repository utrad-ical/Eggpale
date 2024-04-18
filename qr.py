import numpy as np
import scipy

z = np.load('z.npy')

# z(case, element)

assert (z.shape[0] < z.shape[1])

q, r = np.linalg.qr(z.T)

np.save('q.npy', q)

