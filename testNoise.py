from Utils import plotData
from GaussianBoundary import *
import matplotlib.pyplot as plt
import numpy as np

yaxis = np.linspace(-10,10,50)
xaxis = np.linspace(-10,10,50)
x,y = np.meshgrid(xaxis,yaxis)

@np.vectorize
def func(m,n):
    return distanceToCurve([1,0,0],m,n)
z = func(x,y)

#plt.pcolormesh(x,y,z)
#plt.show()

@np.vectorize
def func2(x):
    evalFunction([1,0,0],x)

print(evalFunction([0,0,1],2))
