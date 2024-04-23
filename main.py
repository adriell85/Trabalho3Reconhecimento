import matplotlib
matplotlib.use('TkAgg')
from runs import KNNRuns,DMCRuns,NayveBayesRuns

if __name__ =='__main__':
    for i in range(5):
        NayveBayesRuns(i)
