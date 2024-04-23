import matplotlib
matplotlib.use('TkAgg')
from runs import KNNRuns,DMCRuns,NayveBayesRuns


def main():
    for i in range(5):
        NayveBayesRuns(i)

if __name__ == "__main__":
    main()
