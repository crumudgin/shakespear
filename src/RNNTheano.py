import numpy as np
import operator
import theano
import theano.tensor as T

class RNNTheano:

    def __init__(self, wordDim, hiddenDim=100, bpttTruncate=4):
        self.wordDim = wordDim
        self.hiddenDim = hiddenDim
        self.bpttTruncate = bpttTruncate
        U = np.random.uniform(-np.sqrt(1.0/wordDim), np.sqrt(1.0/wordDim), (hiddenDim,wordDim))
        V = np.random.uniform(-np.sqrt(1.0/hiddenDim), np.sqrt(1.0/hiddenDim), (wordDim, hiddenDim))
        W = np.random.uniform(-np.sqrt(1.0/hiddenDim), np.sqrt(1.0/hiddenDim), (hiddenDim, hiddenDim))
        self.U = theano.shared(name='U', value=U.astype(theano.config.floatX))
        self.V = theano.shared(name='V', value=V.astype(theano.config.floatX))
        self.W = theano.shared(name='W', value=W.astype(theano.config.floatX))

        self.theano = {}
    #     self.__theanoBuild__()
    #
    # def __theanoBuild__(self):
    #     x = T.ivector('x')
    #     y = T.ivector('y')
    #     print(x)
    def forwardProp(self, x):
        print(x)
        X = theano.tensor.ivector()
        f = theano.function([X], X)
        #
        print(f(x))
        # # X = T.imatrix()
        L = len(x)
        l = T.iscalar()
        s = T.dmatrix()
        S = np.zeros((L+1, self.hiddenDim))
        o = np.zeros((L, self.wordDim))
        # # l = T.vector()
        dot = np.exp(self.V.dot(s[l])-np.max(self.V.dot(s[l])))
        # y = T.scalar()
        sFunc = theano.function([s, l, X], np.tanh(self.U[:, X[l]] + self.W.dot(s[l - 1])))
        oFunc = theano.function([s, l], dot/np.sum(np.exp(dot)))
        # for l in np.arange(L):
        for i in np.arange(L):
            S[i] = sFunc(S, i, x)
            o[i] = oFunc(S, i)

    def softmax(self,x):
        xt = np.exp(x - np.max(x))
        return xt / np.sum(xt)
