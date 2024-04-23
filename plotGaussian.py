import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

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