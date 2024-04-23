import numpy as np
from KNN import KNN
from DMC import DMC

from openDatasets import openIrisDataset, openDermatologyDataset,openBreastDataset,openColumnDataset,openArtificialDataset,datasetSplitTrainTest
from plots import confusionMatrix, plotConfusionMatrix,plotDecisionSurface

def KNNRuns(base):
    convertRun = {
        0: openIrisDataset(),
        1: openColumnDataset(),
        2: openArtificialDataset(),
        3: openBreastDataset(),
        4: openDermatologyDataset()
    }
    convertDocName = {
        0: 'Iris',
        1: 'Coluna',
        2: 'Artificial',
        3: 'Breast',
        4: 'Dermatology'

    }

    out = convertRun[base]
    x = out[0]
    y = out[1]
    originalLabels = out[2]
    accuracyList = []

    fileName = "DadosRuns/KNNRuns_{}.txt".format(convertDocName[base])
    with open(fileName, 'w') as arquivo:
        arquivo.write("Execução Iterações KNN {}.\n\n".format(convertDocName[base]))
        for i in range(20):
            print('\nIteração {}\n'.format(i))
            xtrain, ytrain, xtest, ytest = datasetSplitTrainTest(x, y, 80,'KNN',convertDocName[base])
            ypredict = KNN(xtrain, ytrain, xtest, 5)
            confMatrix = confusionMatrix(ytest, ypredict)
            print('Confusion Matrix:\n', confMatrix)
            plotConfusionMatrix(confMatrix,originalLabels,'KNN',i,convertDocName[base])
            accuracy = np.trace(confMatrix) / np.sum(confMatrix)
            print('ACC:', accuracy)
            arquivo.write("ACC: {}\n".format(accuracy))
            arquivo.write("Confusion Matrix: \n {} \n\n".format(confMatrix))
            accuracyList.append(i)
            plotDecisionSurface(xtrain, ytrain,'KNN',i,convertDocName[base])
        print('\nAcurácia média das 20 iterações: {:.2f} ± {:.2f}'.format(np.mean(accuracyList), np.std(accuracyList)))
        arquivo.write(
            '\nAcurácia média das 20 iterações: {:.2f} ± {:.2f}'.format(np.mean(accuracyList), np.std(accuracyList)))


def DMCRuns(base):
    convertRun = {
        0: openIrisDataset(),
        1: openColumnDataset(),
        2: openArtificialDataset(),
        3: openBreastDataset(),
        4: openDermatologyDataset()
    }
    convertDocName = {
        0: 'Iris',
        1: 'Coluna',
        2: 'Artificial',
        3: 'Breast',
        4: 'Dermatology'

    }

    out = convertRun[base]
    x = out[0]
    y = out[1]
    originalLabels = out[2]
    accuracyList = []

    fileName = "DadosRuns/DMCRuns_{}.txt".format(convertDocName[base])
    with open(fileName, 'w') as arquivo:
        arquivo.write("Execução Iterações DMC.\n\n")
        for i in range(20):
            print('\nIteração {}\n'.format(i))
            xtrain, ytrain, xtest, ytest = datasetSplitTrainTest(x, y, 80,'DMC',convertDocName[base])
            ypredict = DMC(xtrain, ytrain, xtest)
            confMatrix = confusionMatrix(ytest, ypredict)
            print('Confusion Matrix:\n', confMatrix)
            plotConfusionMatrix(confMatrix,originalLabels,'DMC',i,convertDocName[base])
            accuracy = np.trace(confMatrix) / np.sum(confMatrix)
            print('ACC:', accuracy)
            arquivo.write("ACC: {}\n".format(accuracy))
            arquivo.write("Confusion Matrix: \n {} \n\n".format(confMatrix))
            accuracyList.append(i)
            plotDecisionSurface(xtrain, ytrain,'DMC',i,convertDocName[base])
        print('\nAcurácia média das 20 iterações: {:.2f} ± {:.2f}'.format(np.mean(accuracyList), np.std(accuracyList)))
        arquivo.write(
            '\nAcurácia média das 20 iterações: {:.2f} ± {:.2f}'.format(np.mean(accuracyList), np.std(accuracyList)))


def NayveBayesRuns(base):
    from NaiveBayes import NaiveBayesClassifier
    convertRun = {
        0: openIrisDataset(),
        1: openColumnDataset(),
        2: openArtificialDataset(),
        3: openBreastDataset(),
        4: openDermatologyDataset()
    }
    convertDocName = {
        0: 'Iris',
        1: 'Coluna',
        2: 'Artificial',
        3: 'Breast',
        4: 'Dermatology'

    }

    out = convertRun[base]
    x = out[0]
    y = out[1]
    originalLabels = out[2]
    accuracyList = []
    fileName = "DadosRuns/NaiveRuns_{}.txt".format(convertDocName[base])
    with open(fileName, 'w') as arquivo:
        arquivo.write("Execução Iterações Naive {}.\n\n".format(convertDocName[base]))
        for i in range(20):
            print('\nIteração {}\n'.format(i))
            xtrain, ytrain, xtest, ytest = datasetSplitTrainTest(x, y, 80,'Naive Bayes Gaussian',convertDocName[base])
            model = NaiveBayesClassifier()
            model.fit(xtrain, ytrain,convertDocName[base],True,i)
            ypredict = model.predict(xtest,convertDocName[base],i)
            confMatrix = confusionMatrix(ytest, ypredict)
            print('Confusion Matrix:\n', confMatrix)
            plotConfusionMatrix(confMatrix,originalLabels,'Naive',i,convertDocName[base])
            accuracy = np.trace(confMatrix) / np.sum(confMatrix)
            print('ACC:', accuracy)
            arquivo.write("ACC: {}\n".format(accuracy))
            arquivo.write("Confusion Matrix: \n {} \n\n".format(confMatrix))
            accuracyList.append(accuracy)
            plotDecisionSurface(xtrain, ytrain,'Naive',i,convertDocName[base])
        print('\nAcurácia média das 20 iterações: {:.2f} ± {:.2f}'.format(np.mean(accuracyList), np.std(accuracyList)))
        arquivo.write(
            '\nAcurácia média das 20 iterações: {:.2f} ± {:.2f}'.format(np.mean(accuracyList), np.std(accuracyList)))