import scipy as sp
import numpy as np
import data as d
import torch
import matplotlib.pyplot as plt
import math

x = 1 + 1;
inputData = d.data.imgCircles(1, 'circles.png');
sumdata = np.sum(inputData, axis = 2);
print(inputData);
print(sumdata);
plt.imshow(sumdata);
plt.gray();
plt.show();