import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

test = [1,1,1,1,2,2,2,2,2,0,0,0,0,0]
print np.argmax(np.bincount(test))
max_vals = np.where(np.bincount(test) == np.amax(np.bincount(test)))[0]
print max_vals
print np.random.choice(max_vals)