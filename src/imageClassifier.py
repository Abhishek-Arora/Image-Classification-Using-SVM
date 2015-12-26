import sys
import os
import itertools
import random
import Image  # PIL
from svmutil import *  # libSVM

# Image data constants
DIMENSION = 32
ROOT_DIR = "../images/"
DAL = "dalmatian"
DOLLAR = "dollar_bill"
PIZZA = "pizza"
BALL = "soccer_ball"
FLOWER = "sunflower"
CLASSES = [DAL, DOLLAR, PIZZA, BALL, FLOWER]

# libsvm constants
LINEAR = 0
RBF = 2

# Other
USE_LINEAR = False
IS_TUNING = False

def main():
    try:
        train, tune, test = getData(IS_TUNING)
        models = getModels(train)
        results = None
        if IS_TUNING:
            print "!!! TUNING MODE !!!"
            results = classify(models, tune)
        else:
            results = classify(models, test)

        print
        totalCount = 0
        totalCorrect = 0
        for clazz in CLASSES:
            count, correct = results[clazz]
            totalCount += count
            totalCorrect += correct
            print "%s %d %d %f" % (clazz, correct, count, (float(correct) / count))
        print "%s %d %d %f" % ("Overall", totalCorrect, totalCount, (float(totalCorrect) / totalCount))

    except Exception as e:
        print e
        return 5

def classify(models, dataSet):
    results = {}
    for trueClazz in CLASSES:
        count = 0
        correct = 0
        for item in dataSet[trueClazz]:
            predClazz, prob = predict(models, item)
            print "%s,%s,%f" % (trueClazz, predClazz, prob)
            count += 1
            if trueClazz == predClazz: correct += 1
        results[trueClazz] = (count, correct)
    return results

def predict(models, item):
    maxProb = 0.0
    bestClass = ""
    for clazz, model in models.iteritems():
        prob = predictSingle(model, item)
        if prob > maxProb:
            maxProb = prob
            bestClass = clazz
    return (bestClass, maxProb)

def predictSingle(model, item):
    output = svm_predict([0], [item], model, "-q -b 1")
    prob = output[2][0][0]
    return prob

def getModels(trainingData):
    models = {}
    param = getParam(USE_LINEAR)
    for c in CLASSES:
        labels, data = getTrainingData(trainingData, c)
        prob = svm_problem(labels, data)
        m = svm_train(prob, param)
        models[c] = m
    return models

def getTrainingData(trainingData, clazz):
    labeledData = getLabeledDataVector(trainingData, clazz, 1)
    negClasses = [c for c in CLASSES if not c == clazz]
    for c in negClasses:
        ld = getLabeledDataVector(trainingData, c, -1)
        labeledData += ld
    random.shuffle(labeledData)
    unzipped = [list(t) for t in zip(*labeledData)]
    labels, data = unzipped[0], unzipped[1]
    return (labels, data)

def getParam(linear = True):
    param = svm_parameter("-q")
    param.probability = 1
    if(linear):
        param.kernel_type = LINEAR
        param.C = .01
    else:
        param.kernel_type = RBF
        param.C = .01
        param.gamma = .00000001
    return param

def getLabeledDataVector(dataset, clazz, label):
    data = dataset[clazz]
    labels = [label] * len(data)
    output = zip(labels, data)
    return output

def getData(generateTuningData):
    trainingData = {}
    tuneData = {}
    testData = {}

    for clazz in CLASSES:
        (train, tune, test) = buildTrainTestVectors(buildImageList(ROOT_DIR + clazz + "/"), generateTuningData)
        trainingData[clazz] = train
        tuneData[clazz] = tune
        testData[clazz] = test

    return (trainingData, tuneData, testData)

def buildImageList(dirName):
    imgs = [Image.open(dirName + fileName).resize((DIMENSION, DIMENSION)) for fileName in os.listdir(dirName)]
    imgs = [list(itertools.chain.from_iterable(img.getdata())) for img in imgs]
    return imgs

def buildTrainTestVectors(imgs, generateTuningData):
    # 70% for training, 30% for test.
    testSplit = int(.7 * len(imgs))
    baseTraining = imgs[:testSplit]
    test = imgs[testSplit:]

    training = None
    tuning = None
    if generateTuningData:
        # 50% of training for true training, 50% for tuning.
        tuneSplit = int(.5 * len(baseTraining))
        training = baseTraining[:tuneSplit]
        tuning = baseTraining[tuneSplit:]
    else:
        training = baseTraining

    return (training, tuning, test)

if __name__ == "__main__":
    sys.exit(main())
