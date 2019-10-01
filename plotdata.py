
import GaussianBoundary as gb
import numpy as np
from Utils import plotData
import matplotlib.pyplot as plt


coVec = [1,0]

tdata = np.array( gb.getPoints(coVec, 1000, 0, 0, -10, 10, -10, 10) )
vdata = np.array( gb.getPoints(coVec, 1000, 0, 0, -10, 10, -10, 10) )

plotData(tdata)