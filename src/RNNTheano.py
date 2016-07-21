import numpy as np
import operator
import theano
import theano.tensor as T

class RNNTheano:

    def __init__(self, wordDim, hiddenDim=100, bpttTruncate=4):
        self.wordDim = wordDim
        self.hiddenDim = hiddenDim
        self.bpttTruncate = bpttTruncate
        self.U = np.random.uniform(-np.sqrt(1.0/wordDim), np.sqrt(1.0/wordDim), (hiddenDim,wordDim))
        self.V = np.random.uniform(-np.sqrt(1.0/hiddenDim), np.sqrt(1.0/hiddenDim), (wordDim, hiddenDim))
        self.W = np.random.uniform(-np.sqrt(1.0/hiddenDim), np.sqrt(1.0/hiddenDim), (hiddenDim, hiddenDim))

        self.theano = {}
        self.__theanoBuild__()

    def __theanoBuild(self):
        U,V,W = self.U, self.V, self.W
