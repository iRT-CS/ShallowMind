import json
import numpy as np

with open('dataset.json') as f:
  data = json.load(f)

#print(data)
trainingvals = data['trainingValues']
xarr = []
for d in trainingvals[0]:
    xarr.append(float(d['$numberDouble']))

yarr = []
for d in trainingvals[1]:
    yarr.append(float(d['$numberDouble']))

classification = []
for d in trainingvals[2]:
    classification.append(float(d['$numberInt']))

tdata = np.array([xarr,yarr,classification])
tdata = tdata.T
