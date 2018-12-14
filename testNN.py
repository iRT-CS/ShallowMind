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
    training, test = data
    training = np.array(training)
    vErrorConsec = 0
    cont = True
    tError = []
    vError = []

    #boolean consec

    while(cont):
        #train (1) epochs
        nn.fit(x=training, y=test, batch_size=100, epochs=1, verbose=1)

        # call evaluate - record test & validation error
        stats = model.evaluate(x=training, y=testing, batch_size=100, epochs=1, verbose=1)

        # record test error
        tError.append(stats[0])
        # record validation error
        vError.append(stats[2])

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
