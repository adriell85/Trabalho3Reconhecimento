import numpy as np

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

            lineSplited = int(line.split(',')[-1])-1

            features = [np.nan if feature == '?' else float(feature) for feature in features[:-1]]
            x.append(features)
            y.append(lineSplited)

    x = np.array(x, dtype=np.float64)
    for i in range(x.shape[1]):
        column_mean = np.nanmean(x[:, i])
        np.place(x[:, i], np.isnan(x[:, i]), column_mean)

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

