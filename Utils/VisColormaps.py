from matplotlib.colors import ListedColormap
import numpy as np
import os
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
from matplotlib import cm

def plotColorbar(colormap):
    fig = plt.figure()
    ax = fig.add_subplot()
    fig.colorbar(cm.ScalarMappable(cmap=colormap))
    plt.show()
# makes a more defined split between the colors
def createColormap():
    orig = cm.get_cmap('RdBu', 256)
    colors = np.array(orig(np.linspace(0, 1, 256)))
    length = len(colors)
    split = round(length * .4)
    top_40 = colors[:split+1]
    bottom_40 = colors[-split:]
    full_cmp = np.vstack((top_40, bottom_40))
    cmp = ListedColormap(full_cmp, "RedBlue")
    return cmp

# def showColormap():
#     colormap = createColormap()
#     plotColorbar(colormap=colormap)

# showColormap()