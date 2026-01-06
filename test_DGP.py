

from __future__ import absolute_import, division, print_function

import argparse
from pyro.contrib.examples.util import get_data_loader
from models.DGP import DeepGP
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import time
import sys
import numpy as np
import torch


def test(test_loader, gpmodule):
    test = test_loader.dataset.data.reshape(-1, 4096).float() / 255
    print(list(test.size()))  # [25110, 4096]
    testy = test_loader.dataset.targets
    print(testy.shape[:-1])
    pred = gpmodule(test)
    np.savetxt('Y_test.csv', testy, delimiter=',')
    np.savetxt('Y_test_pred.csv', pred, delimiter=',',  fmt = "%s")
    outputfile = open("RESULT.txt", "a")
    sys.stdout = outputfile
    print("acc:", accuracy_score(testy, pred))
    print("pre", precision_score(testy, pred, average='macro'))
    print("recall", recall_score(testy, pred, average='micro'))
    print("F1", f1_score(testy, pred, average='macro'))



def main(args):

    train_loader = get_data_loader(dataset_name='MNIST',
                                   data_dir='~/.data',
                                   batch_size=100,
                                   is_training_set=True,
                                   shuffle=True)

    test_loader = get_data_loader(dataset_name='MNIST',
                                  data_dir='~/.data',
                                  batch_size=100,
                                  is_training_set=False,
                                  shuffle=False)


    X = train_loader.dataset.data.reshape(-1, 4096).float() / 255
    print(list(X.size()))#[25110, 4096]
    y = train_loader.dataset.targets
    print(y.shape[:-1])

    deepgp = DeepGP(X, y, num_classes=81)
    time_start = time.time()
    deepgp.train(args.num_epochs, args.num_iters, args.batch_size, args.learning_rate)
    time_end = time.time()
    cost = time_end - time_start
    print('totally cost', time_end - time_start)
    test(test_loader, deepgp)
    f1 = open("RESULT.txt", 'a')
    f1.write(str(cost))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Deep Gaussian Processes on MNIST")

    parser.add_argument("-n", "--num-epochs", default=200, type=int)
    parser.add_argument("-t", "--num-iters", default=99, type=int)
    parser.add_argument("-b", "--batch-size", default=256, type=int)
    parser.add_argument("-lr", "--learning-rate", default=0.01, type=float)

    args = parser.parse_args()

    main(args)


# In[ ]:



