from Utils import iterate

MAX_NODES = 6
MAX_LAYERS = 4

iter = iterate([],MAX_LAYERS,MAX_NODES)

while(iter != -1):
    print("iter = " + str(iter))
    iter = iterate(iter,MAX_LAYERS,MAX_NODES)
