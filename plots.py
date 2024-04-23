import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
import matplotlib
matplotlib.use('TkAgg')
from NaiveBayesClassifier import NaiveBayesClassifier
from KNN import KNN
from DMC import DMC
from scipy.stats import multivariate_normal

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


def plotGaussianDistribution(means, covariances, classes, featureIndices=(0, 1), gridRange=(-3, 3), resolution=0.1):

    f1, f2 = featureIndices
    x, y = np.mgrid[gridRange[0]:gridRange[1]:resolution, gridRange[0]:gridRange[1]:resolution]
    pos = np.dstack((x, y))

    fig, ax = plt.subplots()

    for i, c in enumerate(classes):
        mean = means[i][[f1, f2]]
        covariance = covariances[i][[f1, f2], [f1, f2]]
        rv = multivariate_normal(mean, covariance)
        ax.contourf(x, y, rv.pdf(pos), levels=100, cmap='Blues', alpha=0.5)
        ax.set_title(f'Multivariate Gaussian Distribution - Features {f1} and {f2}')
        ax.set_xlabel(f'Feature {f1}')
        ax.set_ylabel(f'Feature {f2}')

    plt.show()



def plotGaussianDistribution3d(baseName,iteration,means, covariances, classes, featureIndices=(0, 1), gridRange=(-0.3, 0.3), resolution=0.1):
    atributesCombinationArtificial = [
        [0, 1]
    ]
    atributesCombinationIris = [
        [0, 1],
        [0, 2],
        [0, 3],
        [1, 2],
        [1, 3],
        [2, 3]
    ]
    atributesCombinationFree = [
        [0, 1],
        [0, 4],
        [0, 5],
        [2, 3],
        [3, 4],
        [4, 5]
    ]
    if (baseName == 'Iris'):
        atributesCombination = atributesCombinationIris
    elif (baseName == 'Artificial'):
        atributesCombination = atributesCombinationArtificial
    else:
        atributesCombination = atributesCombinationFree
    for ind in atributesCombination:
        f1, f2 = ind
        x, y = np.mgrid[gridRange[0]:gridRange[1]:resolution, gridRange[0]:gridRange[1]:resolution]
        pos = np.dstack((x, y))

        fig2 = plt.figure()


        for i, c in enumerate(classes):
            ax = fig2.add_subplot(111, projection='3d')
            mean = means[i][[f1, f2]]
            covariance = covariances[i][[f1, f2], [f1, f2]]

            rv = multivariate_normal(mean=mean, cov=covariance, allow_singular=True)
            z = rv.pdf(pos)

            ax.plot_surface(x, y, z, cmap='cividis', edgecolor='none', alpha=0.5)
            ax.set_title(f'Multivariate Gaussian Distribution - Features {f1} and {f2} base {baseName}')
            ax.set_xlabel(f'Feature {f1}')
            ax.set_ylabel(f'Feature {f2}')
            ax.set_zlabel('Probability')
            plt.savefig('Resultados_Naive/{}/Gaussiana_Base_{}_features_{}_classe_{}_iteracao_{}.png'.format(baseName,baseName,ind,i,iteration))

        # plt.show()