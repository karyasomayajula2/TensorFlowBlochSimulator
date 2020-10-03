import scipy as sp
import numpy as np
import data as d
import torch
import matplotlib.pyplot as plt
import math

inputData = d.data.imgCircles(1);
sumdata = np.sum(inputData, axis = 2);
print(inputData);
print(sumdata);
plt.imshow(inputData);
plt.show();