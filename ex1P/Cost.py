import numpy as np
import h5py
f = h5py.File('D:\Users\Rares\Documents\Machine_Learning\machine-learning-ex1\ex1\data1.mat','r')
data = f.get('data/variable1')
data = np.array(data) # For converting to numpy array
print(data);