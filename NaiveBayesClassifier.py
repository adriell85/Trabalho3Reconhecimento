from scipy.stats import multivariate_normal
import numpy as np
import matplotlib.pyplot as plt


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

def regularizeCovariance(covariance, alpha=1e-5):

    regularizedCovariance = covariance + alpha * np.eye(covariance.shape[0])
    return regularizedCovariance

def plotGaussianDistribution3d(baseName,means, covariances, classes, featureIndices=(0, 1), gridRange=(-1, 1), resolution=0.1,):

    atributesCombination3Features = [
        [0, 1],
        [0, 2],
        [0, 3],
        [1, 2],
        [1, 3],
        [2, 3]
    ]
    atributesCombination5Features = [
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4],
        [0, 5],
        [1, 2],
        [1, 3],
        [1, 4],
        [1, 5],
        [2, 3],
        [2, 4],
        [2, 5],
        [3, 4],
        [3, 5],
        [4, 5]
    ]
    for ind in atributesCombination5Features:
        f1, f2 = ind
        x, y = np.mgrid[gridRange[0]:gridRange[1]:resolution, gridRange[0]:gridRange[1]:resolution]
        pos = np.dstack((x, y))

        fig2 = plt.figure()
        ax = fig2.add_subplot(111, projection='3d')

        for i, c in enumerate(classes):
            mean = means[i][[f1, f2]]
            covariance = covariances[i][[f1, f2], [f1, f2]]

            rv = multivariate_normal(mean=mean, cov=covariance, allow_singular=True)
            z = rv.pdf(pos)

            ax.plot_surface(x, y, z, cmap='viridis', edgecolor='none', alpha=0.5)
            ax.set_title(f'Multivariate Gaussian Distribution - Features {f1} and {f2} base {baseName}')
            ax.set_xlabel(f'Feature {f1}')
            ax.set_ylabel(f'Feature {f2}')
            ax.set_zlabel('Probability')
            plt.savefig('Resultados_Naive/{}/Gaussiana_Base_{}_features_{}.png'.format(baseName,baseName,ind))

        plt.show()



# def naiveBayesGaussianMultivar(xTrain, yTrain, xTest,baseName):
#     xTrain = np.array(xTrain)
#     yTrain = np.array(yTrain)
#     xTest = np.array(xTest)
#     classes = np.unique(yTrain)
#     nClasses = len(classes)
#     nFeatures = xTrain.shape[1]
#     priors = np.zeros(nClasses)
#     means = np.zeros((nClasses, nFeatures))
#     covariances = np.zeros((nClasses, nFeatures, nFeatures))
#
#     for i, c in enumerate(classes):
#         xC = xTrain[yTrain == c]
#         priors[i] = xC.shape[0] / xTrain.shape[0]
#         means[i] = np.mean(xC, axis=0)
#         covariances[i] = np.cov(xC, rowvar=False)+ np.eye(nFeatures) * 1e-4
#
#
#     # plotGaussianDistribution3d(baseName, means, covariances, classes, featureIndices=(1, 2))
#     def multivariateGaussianPdf(x, mean, covariance):
#         detCov = np.linalg.det(covariance)
#         invCov = np.linalg.inv(covariance)
#         numerator = np.exp(-0.5 * (x - mean) @ invCov @ (x - mean).T)
#         denominator = np.sqrt((2 * np.pi) ** nFeatures * detCov)
#         return numerator / denominator
#
#     def classify(x):
#         posteriors = np.zeros(nClasses)
#         for i in range(nClasses):
#             likelihood = multivariateGaussianPdf(x, means[i], covariances[i])
#             posteriors[i] = likelihood * priors[i]
#         return classes[np.argmax(posteriors)]
#
#     predictions = np.array([classify(x) for x in xTest])
#
#     return predictions

class NaiveBayesClassifier:
    def fit(self, xtrain, ytrain,baseName):
        xtrain = np.array(xtrain)
        nSamples,nFeatures = xtrain.shape
        self.classes = np.unique(ytrain)
        nClasses = len(self.classes)
        # print('xtrain',xtrain)

        self.mean = []
        self.variance = []
        self.priorProb = []

        for _class in self.classes:
            _class = xtrain[ytrain==_class]
            self.mean.append(np.mean(_class,axis=0))
            self.variance.append(np.var(_class,axis=0))
            self.priorProb.append(_class.shape[0]/nSamples)
        self.mean=np.array(self.mean)
        self.variance=np.array(self.variance)
        self.priorProb=np.array(self.priorProb)

        plotGaussianDistribution3d(baseName, self.mean, self.variance, self.classes, featureIndices=(1, 2))

    def predict(self,xtest):
        # ypredicted = precit
        predicts=[]
        for xsample in xtest:
            posteriorsPros = []
            # print(self.classes)
            for i,c in enumerate(self.classes):
                priorprobability = np.log(self.priorProb[i])
                conditionalClass = np.sum(np.log(self._pdf(i,xsample)))
                posteriorProbability = priorprobability + conditionalClass
                posteriorsPros.append(posteriorProbability)
            predicts.append(np.argmax(posteriorsPros))
        return np.array(predicts)

    def _pdf(self,iClass,sample):
        mean = self.mean[iClass]
        variance = self.variance[iClass]
        # Adicionando um pequeno valor à variância para evitar divisão por zero
        epsilon = 1e-7
        variance+= epsilon
        numerator = np.exp( - ( sample- mean ) ** 2 / (2 * variance) )
        denominator = np.sqrt( 2 * np.pi * variance )
        return numerator / denominator


