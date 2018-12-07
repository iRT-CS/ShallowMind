from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
import keras
from Utils import iterate

# Continuously runs epochs on neural net with given data points until error is minimized
# nn: compiled neural net
# data: list of data points
#
def test(nn, tdata, vdata):
    vErrorConsec = 0
    cont = True
    tError = []
    vError = []

    while(cont):
        #train (1) epochs


        #record test error
        #record validation error

        #terminate if error is under 20% and vErrorConsec is greater than 5
        if(len(vError) > 1 and vError[len(vError)-1] < vError[len(vError)-2]):
            vErrorConsec += 1
        if(vError[len(vError)-1] < 0.2 and vErrorConsec > 5):
            break

ids = []
id = iterate(1, 4, 6)
newid = 0
while(id != -1):
    print(id)
    newid = iterate(id, 4,6)
    id = newid
