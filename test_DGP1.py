from __future__ import absolute_import, division, print_function
import argparse
import numpy as np
from models.EK_DGP import DeepGP
import torch
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import time
import sys


def test(X_test,y_test, gpmodule):
    pred = gpmodule(X_test)
    sum = 0.0
    for i in range(2999):
        if (pred[i] == y_test[i]):
            sum = sum + 1
    np.savetxt('Y_test.csv', y_test, delimiter=',')
    np.savetxt('Y_test_pred.csv', pred, delimiter=',')
    outputfile = open("RESULT.txt", "a")
    sys.stdout = outputfile
    print("\nTest set: Accuracy: {}/{} ({:.2f}%)\n".format(sum, 2999, 100. * sum / 2999))
    print("acc:", accuracy_score(y_test, pred))
    print("pre", precision_score(y_test, pred, average='macro'))
    print("recall", recall_score(y_test, pred, average='micro'))
    print("F1", f1_score(y_test, pred, average='macro'))




def main(args):
    # data_train = np.loadtxt(open("C:/Users/zm/Desktop/第一章特征/融合特征/Sedensenet50LBPLDAtrain_features.csv", encoding='gb18030', errors="ignore"), delimiter=",", skiprows=0)
    # data_test = np.loadtxt(open("C:/Users/zm/Desktop/第一章特征/融合特征/Sedensenet50LBPLDAtest_features.csv", encoding='gb18030', errors="ignore"), delimiter=",", skiprows=0)
    data_train = np.loadtxt(open("C:/Users/zm/Desktop/寒假论文/第一章特征/SEDensenet/Oracletrain_features.csv", encoding='gb18030', errors="ignore"), delimiter=",", skiprows=0)
    data_test = np.loadtxt(open("C:/Users/zm/Desktop/寒假论文/第一章特征/SEDensenet/Oracletest_features.csv", encoding='gb18030', errors="ignore"), delimiter=",", skiprows=0)
    X_train, y_train = data_train[:, :-1], data_train[:, -1]
    X2, y2 = data_test[:, :-1], data_test[:, -1]
    X = torch.from_numpy(X_train)
    y = torch.from_numpy(y_train)
    X_test = torch.from_numpy(X2)
    y_test = torch.from_numpy(y2)
    deepgp = DeepGP(X, y, num_classes=10)
    time_start = time.time()
    deepgp.train(args.num_epochs, args.num_iters, args.batch_size, args.learning_rate)
    time_end = time.time()
    cost = time_end - time_start
    print('totally cost', time_end - time_start)
    test(X_test, y_test, deepgp)
    f1 = open("RESULT.txt", 'a')
    f1.write(str(cost))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Deep Gaussian Processes")
    parser.add_argument("-n", "--num-epochs", default=100, type=int)
    parser.add_argument("-t", "--num-iters", default=118, type=int)
    parser.add_argument("-b", "--batch-size", default=256, type=int)
    parser.add_argument("-lr", "--learning-rate", default=0.01, type=float)
    args = parser.parse_args()
    main(args)




