# -*- coding: utf-8 -*-
# File: format.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import numpy as np
import six
from six.moves import range
import os

class SVMLightData:
    """ Read X,y from a svmlight file, and produce [X_i, y_i] pairs. """

    def __init__(self, filename, shuffle=True):
        """
        Args:
            filename (str): input file
            shuffle (bool): shuffle the data
        """
        import sklearn.datasets  # noqa
        self.X, self.y = sklearn.datasets.load_svmlight_file(filename)
        self.X = np.asarray(self.X.todense())
        self.shuffle = shuffle

    def size(self):
        return len(self.y)

    def get_data(self):
        idxs = np.arange(self.size())
        if self.shuffle:
            self.rng.shuffle(idxs)
        for id in idxs:
            yield [self.X[id, :], self.y[id]]



