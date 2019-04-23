import numpy as np
from scipy.optimize import fmin_cobyla

def pol2car(r,theta):
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    return x,y
def car2pol(x,y):
    r = np.sqrt((x**2)+(y**2))
    theta = np.arctan(y/x)
    return r,theta

def evalEllipse(a,b,theta):
    num = a*b
    den = np.sqrt(((b*np.cos(theta))**2)+((a*np.sin(theta))**2))
    return num/den
def getEDistance(a,b,x,y):
    r,theta = car2pol(x,y)
    eTheta = fmin_cobyla(lambda x: abs(r-evalEllipse(a,b,x[0])),x0 = [theta,theta],cons = [lambda x: x[0],lambda x: 6.2832-x[0]], rhoend = 1e-3)[1]
    eR = evalEllipse(a,b,eTheta)
    eX, eY = pol2car(eR,eTheta)
    dX = x-eX
    dY = y-eY
    dist = np.sqrt((dX**2)+(dY**2))
    return dist


print(getEDistance(1,1,2,0))
