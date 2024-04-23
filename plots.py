import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
import matplotlib
matplotlib.use('TkAgg')
from NaiveBayes import NaiveBayesClassifier
from KNN import KNN
from DMC import DMC


def confusionMatrix(y_true, y_pred):
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    num_classes = max(max(y_true), max(y_pred)) + 1
    conf_matrix = np.zeros((num_classes, num_classes), dtype=int)

    for true, pred in zip(y_true, y_pred):
        conf_matrix[true][pred] += 1

    return conf_matrix

def plotConfusionMatrix(conf_matrix, class_names,classifierName,i,datasetName):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Greens", xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('**True Label**')
    plt.xlabel('**Predicted Label**')
    plt.title('Confusion Matrix')
    plt.savefig('Resultados_{}/{}/Matriz_de_Confusao_base_{}_Iteracao_{}.png'.format(classifierName,datasetName,datasetName,i))




def plotDecisionSurface(xtrain,ytrain,classifierName,i,datasetName):
    atributesCombinationArtificial=[
        [0,1]
    ]
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
    elif(datasetName=='Artificial'):
        atributesCombination = atributesCombinationArtificial
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
            Z = model.predict(matrix,datasetName,i,True)
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





