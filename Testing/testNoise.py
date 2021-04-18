from Utils import plotData
from Datasets.GaussianBoundary import *
import matplotlib.pyplot as plt
import numpy as np
import math
from Utils.conicDistances import getEDistance,car2pol

yaxis = np.linspace(-10,10,50)
xaxis = np.linspace(-10,10,50)
x,y = np.meshgrid(xaxis,yaxis)

@np.vectorize
def func(m,n):
    return getEDistance([10,5],m,n)
    #return car2pol(m,n)[1]
z = func(x,y)

plt.pcolormesh(x,y,z)
plt.show()
