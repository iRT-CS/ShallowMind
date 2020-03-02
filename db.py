from pymongo import MongoClient
client = MongoClient('shallowmind.pingry.org', 27017)
db = client["ShallowMind"]

def createExperimentsDocument(
#links to NeuralNets collection
neuralNetUniqueID,
#see createNeuralNetsDocument
neuralNetHiddenStructure,
#see createNeuralNetsDocument
inputShape,
#see createNeuralNetsDocument
outputShape,
#links to Dataset collection
datasetUniqueID,
#training accuracy at each epoch
trainingLossOverTime,
#validation accuracy at each epoch
validationLossOverTime,
#dictionary whose keys are stopping criterion
stoppingCriterionDictionary
#i.e. stoppingCriterionDictionary["Stop when the validation error increases for 5 consecutive epochs"]
#the value will be another dictionary which will have three keys
#"Final validation error","Final training error", and "Final weights"
):
    collection = db.Experiments
    document = {
    "neuralNetUniqueID": neuralNetUniqueID,
    "neuralNetHiddenStructure": neuralNetHiddenStructure,
    "inputShape": inputShape,
    "outputShape": outputShape,
    "datasetUniqueID": datasetUniqueID,
    "trainingLossOverTime": trainingLossOverTime,
    "validationLossOverTime": validationLossOverTime,
    "stoppingCriterionDictionary": stoppingCriterionDictionary
    }
    return collection.insert_one(document).inserted_id



def createNeuralNetsDocument(
#array of hidden layer sizes
neuralNetHiddenStructure,
#size of input layer
inputShape,
#size of output later
outputShape,
#list of numpy arrays containing initial weights
initialWeights,
#function used to initialize weights (i.e. glorot, he et al)
initializationFunction,
#function used to calculate activation in each neuron (i.e. sigmoid, tanh)
activationFunction
):
    if (neuralNetHiddenStructure == [2]):
        print( "ERROR FOUND")
    if (neuralNetHiddenStructure == [1]):
        print( "CORRECT THING HAPPENED")
    collection = db.NeuralNets
    document = {
    "neuralNetHiddenStructure": neuralNetHiddenStructure,
    "inputShape": inputShape,
    "outputShape": outputShape,
    "initialWeights": initialWeights,
    "initializationFunction": initializationFunction,
    "activationFunction": activationFunction
    }
    return collection.insert_one(document).inserted_id

def createDatasetsDocument(
#array of polynomial coefficient. Higher order coefficient first (i.e. x^2 + 2x + 3 is [1,2,3])
polynomial,
#array that contains the constants describing the noise distribution [peak noise value, spread of the noise (sigma)]
noiseDistribution,
#array that describes the range of data points [x minimum, x maximum, y minimum, y maximum] (i.e. all data point obey -10<x<10 and 0<y<10 [-10,10,0,10])
dataRange,
#1000 values used to train the neural nets
trainingValues,
#1000 different values used to test the neural nets
testValues

#NOTE: The data points will be stored in an array of shape (3,n) where n is the number of points
#In this array, index 0 contains X values, 1 contains Y values, and 2 contains classifications

):
    collection = db.Datasets
    document = {
    "polynomial": polynomial,
    "noiseDistribution": noiseDistribution,
    "range": dataRange,
    "trainingValues":trainingValues.T.tolist(),
    "testValues": testValues.T.tolist()
    }
    return collection.insert_one(document).inserted_id
