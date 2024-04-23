import numpy as np
import pandas as pd

def gerar_dataset():
    num_amostras_classe_0 = 8
    num_amostras_classe_1 = 8
    num_amostras_classe_2 = 7


    x1_classe_0 = np.random.uniform(0, 5, num_amostras_classe_0)
    x2_classe_0 = np.random.uniform(0, 2.5, num_amostras_classe_0)
    labels_classe_0 = np.zeros(num_amostras_classe_0, dtype=int)


    x1_classe_1 = np.random.uniform(15, 20, num_amostras_classe_1)
    x2_classe_1 = np.random.uniform(0, 5, num_amostras_classe_1)
    labels_classe_1 = np.ones(num_amostras_classe_1, dtype=int)


    x1_classe_2 = np.random.uniform(0, 2.5, num_amostras_classe_2)
    x2_classe_2 = np.random.uniform(10, 12.5, num_amostras_classe_2)
    labels_classe_2 = np.full(num_amostras_classe_2, 2, dtype=int)


    x1 = np.concatenate((x1_classe_0, x1_classe_1, x1_classe_2))
    x2 = np.concatenate((x2_classe_0, x2_classe_1, x2_classe_2))
    labels = np.concatenate((labels_classe_0, labels_classe_1, labels_classe_2))


    dataset = pd.DataFrame({
        'x1': x1,
        'x2': x2,
        'Classe': labels
    })

    return dataset


dataset_exemplo = gerar_dataset()

dataset_exemplo.head()

print(dataset_exemplo)