import numpy as np
from numpy.polynomial.polynomial import *
#Find the vector that corresponds to the derivative of the square of the distance function

def distanceToCurve(curve,x,y):
    #Reverse the order so it makes sense
    poly = curve[::-1]
    xterm = polysub([0,1],[x])
    yterm = polysub(poly,[y])
    xterm = polypow(xterm,2)
    yterm = polypow(yterm,2)
    dsquared = polyadd(xterm,yterm)
    deriv = polyder(dsquared)
    roots = polyroots(deriv).real

    currentMin = np.sqrt(polyval(x,dsquared).real)
    for root in np.nditer(roots):
        testVal = np.sqrt(polyval(root,dsquared).real)
        if(currentMin > testVal):
            currentMin = testVal
    return currentMin
        
        
    
    
    
