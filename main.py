import pandas as pd
import numpy as np
from KNN import KNN
from DMC import DMC
from NaiveBayesClassifier import NaiveBayesClassifier
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from numba import njit



def openIrisDataset():
    x = []
    y = []
    originalLabel = []
    ConvertLabel = {
        'Iris-setosa':0,
        'Iris-versicolor':1,
        'Iris-virginica':2
    }
    with open("bases/iris/iris.data") as file:
        for line in file:
            label = ConvertLabel[str(line.split(',')[-1].strip())]
            originalLabel.append(str(line.split(',')[-1].strip()))
            y.append(label)
            x.append([float(feature) for feature in line.split(',')[0:4]])
    print('IRIS Dataset Opened!')
    return [x,y,np.unique(originalLabel)]


def openColumnDataset():
    x = []
    y = []
    originalLabel = []
    ConvertLabel = {
        'DH': 0,
        'SL': 1,
        'NO': 2
    }
    with open("bases/vertebral+column/column_3C.dat") as file:
        for line in file:
            label = ConvertLabel[str(line.split(' ')[-1].strip())]
            originalLabel.append(str(line.split(' ')[-1].strip()))
            y.append(label)
            x.append([float(feature) for feature in line.split(' ')[0:6]])
        newX = normalizeColumns(x).tolist()
    print('Column Dataset Opened!')

    return [newX, y, np.unique(originalLabel)]


def openArtificialDataset():
    x = []
    y = []
    originalLabel = []
    with open("bases/artificial/artificial.txt") as file:
        for line in file:
            splt = line.split(' ')[-1]
            label = int(line.split(' ')[-1].strip())
            y.append(label)
            x.append([float(feature) for feature in line.split(' ')[0:2]])

    print('Column Dataset Opened!')

    return [x, y, np.unique(originalLabel)]



def openBreastDataset():
    x = []
    y = []
    originalLabel = []
    ConvertLabel = {
        'M': 0,
        'B': 1,
    }
    with open("bases/Breast Cancer/wdbc.data") as file:
        for line in file:
            label = ConvertLabel[(line.split(',')[1])]
            originalLabel.append(str(line.split(',')[1]))
            y.append(label)
            x.append([(feature) for feature in line.split(',')[2:]])
        newX = normalizeColumns(np.float64(x)).tolist()
    print('Breast Cancer Dataset Opened!')

    return [newX, y, np.unique(originalLabel)]

def openDermatologyDataset():
    x = []
    y = []
    originalLabel = []
    with open("bases/dermatology/dermatology.data") as file:
        for line in file:
            features = line.split(',')

            lineSplited = int(line.split(',')[-1])
            # Convertendo '?' para np.nan (valor ausente do NumPy)
            features = [np.nan if feature == '?' else float(feature) for feature in features[:-1]]
            x.append(features)
            y.append(lineSplited)  # Supondo que a última coluna seja o rótulo

    # Convertendo x para um array do NumPy para facilitar manipulações
    x = np.array(x, dtype=np.float64)

    # Substituindo valores ausentes pela média da coluna (ignorando valores NaN na média)
    for i in range(x.shape[1]):
        column_mean = np.nanmean(x[:, i])
        np.place(x[:, i], np.isnan(x[:, i]), column_mean)

    # Normalização das colunas
    newX = normalizeColumns(x).tolist()

    print('Breast Cancer Dataset Opened!')

    return [newX, y, np.unique(originalLabel)]








def split_data_randomly(data, percentage):
    if percentage < 0 or percentage > 100:
        raise ValueError("A porcentagem deve estar entre 0 e 100.")
    total_data = len(data)
    size_first_group = int(total_data * (percentage / 100))
    indices = np.random.permutation(total_data)
    first_group_indices = indices[:size_first_group]
    second_group_indices = indices[size_first_group:]
    first_group = []
    second_group = []
    for indice in first_group_indices:
        first_group.append(data[int(indice)])
    for indice in second_group_indices:
        second_group.append(data[int(indice)])

    return first_group, second_group

def datasetSplitTrainTest(x,y,percentageTrain,labelClassifier,labelDataset):

    dataToSplit = [[x,y] for x,y in zip(x,y)]

    percentageTrain = 100 - percentageTrain  # porcentagem de treino

    group1, group2 = split_data_randomly(dataToSplit, percentageTrain)

    xtrain, ytrain = zip(*[(group[0],group[1]) for group in group2])
    xtest, ytest = zip(*[(group[0], group[1]) for group in group1])

    return xtrain, ytrain, xtest, ytest
def normalizeColumns(dataset):
    dataset =np.array(dataset)
    X_min = dataset.min(axis=0)
    X_max = dataset.max(axis=0)

    # Normalizando o dataset
    datasetNormalized = (dataset - X_min) / (X_max - X_min)
    return datasetNormalized



def confusionMatrix(y_true, y_pred):
    num_classes = max(max(y_true), max(y_pred)) + 1
    conf_matrix = np.zeros((num_classes, num_classes), dtype=int)

    for true, pred in zip(y_true, y_pred):
        conf_matrix[true][pred] += 1

    return conf_matrix

def plotConfusionMatrix(conf_matrix, class_names,classifierName,i,datasetName):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('**True Label**')
    plt.xlabel('**Predicted Label**')
    plt.title('Confusion Matrix')
    plt.savefig('Resultados_{}/{}/Matriz_de_Confusao_base_{}_Iteracao_{}.png'.format(classifierName,datasetName,datasetName,i))




def plotDecisionSurface(xtrain,ytrain,classifierName,i,datasetName):
    atributesCombination=[]
    atributesCombinationIris = [
        [0,1],
        [0,2],
        [0,3],
        [1,2],
        [1,3],
        [2,3]
    ]
    atributesCombinationFree = [
        [0, 1],
        [0, 4],
        [0, 5],
        [2, 3],
        [3, 4],
        [4, 5]
    ]
    if(datasetName=='Iris'):
        atributesCombination = atributesCombinationIris
    else:
        atributesCombination = atributesCombinationFree

    for z in atributesCombination:
        xtrainSelected = np.array(xtrain)
        xtrainSelected = xtrainSelected[:, z]
        xtrainSelected = xtrainSelected.tolist()
        xtrainSelected = tuple(xtrainSelected)
        x_min = min([x[0] for x in xtrainSelected]) - 1
        x_max = max([x[0] for x in xtrainSelected]) + 1
        y_min = min([x[1] for x in xtrainSelected]) - 1
        y_max = max([x[1] for x in xtrainSelected]) + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                             np.arange(y_min, y_max, 0.1))
        matrix = np.c_[xx.ravel(), yy.ravel()]
        matrix = matrix.tolist()
        matrix = tuple(matrix)
        if(classifierName=='KNN'):
            Z = KNN(xtrainSelected, ytrain, matrix, k=3)
        elif(classifierName=='DMC'):
            Z = DMC(xtrainSelected, ytrain, matrix)
        else:
            model = NaiveBayesClassifier()
            model.fit(xtrainSelected,ytrain,datasetName,False,i)
            Z = model.predict(matrix,datasetName,i)
        Z = np.array(Z)
        Z = Z.reshape(xx.shape)
        fig, ax = plt.subplots()
        colors = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
        plt.contourf(xx, yy, Z, alpha=0.4, cmap=colors)
        x_vals = [sample[0] for sample in xtrainSelected]
        y_vals = [sample[1] for sample in xtrainSelected]
        plt.scatter(x_vals, y_vals, c=ytrain, s=20, edgecolor='k', cmap=colors)

        plt.title('Superfície de Decisão do {} base {}'.format(classifierName,datasetName))

        plt.xlabel('Atributo 1')
        plt.ylabel('Atributo 2')
        fig.savefig('Resultados_{}/{}/Superficie_de_decisao_base_{}_Atributos_{}_Iteracao_{}.png'.format(classifierName,datasetName,datasetName,z,i))
        # plt.show()




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

    fileName = "KNNRuns_{}.txt".format(convertDocName[base])
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

    fileName = "DMCRuns_{}.txt".format(convertDocName[base])
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
    fileName = "NaiveRuns_{}.txt".format(convertDocName[base])
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


if __name__ =='__main__':
    NayveBayesRuns(1)
    # NayveBayesRuns(2)
    # NayveBayesRuns(0)
    # NayveBayesRuns(4)
    # DMCRuns(0)
    # DMCRuns(1)
    # DMCRuns(2)
    # DMCRuns(3)
    # DMCRuns(4)
    # KNNRuns(0)
    # KNNRuns(1)
    # KNNRuns(2)
    # KNNRuns(3)
    # KNNRuns(4)