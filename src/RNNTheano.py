import numpy as np
import operator
import theano
import theano.tensor as T
import json

class RNNTheano:

    def __init__(self, wordDim, hiddenDim=100, bpttTruncate=4):
        self.wordDim = wordDim
        self.hiddenDim = hiddenDim
        self.bpttTruncate = bpttTruncate
        # U = np.random.uniform(-np.sqrt(1.0/wordDim), np.sqrt(1.0/wordDim), (hiddenDim,wordDim))
        # V = np.random.uniform(-np.sqrt(1.0/hiddenDim), np.sqrt(1.0/hiddenDim), (wordDim, hiddenDim))
        # W = np.random.uniform(-np.sqrt(1.0/hiddenDim), np.sqrt(1.0/hiddenDim), (hiddenDim, hiddenDim))
        U = self.fromFile("U.txt")
        V = self.fromFile("V.txt")
        W = self.fromFile("W.txt")
        self.U = theano.shared(name='U', value=U.astype(theano.config.floatX))
        self.V = theano.shared(name='V', value=V.astype(theano.config.floatX))
        self.W = theano.shared(name='W', value=W.astype(theano.config.floatX))
        # self.toFile(self.U.get_value().tolist(), "U.txt")
        # self.toFile(self.V.get_value().tolist(), "V.txt")
        # self.toFile(self.W.get_value().tolist(), "W.txt")
        self.__theanoBuild__()



    def __theanoBuild__(self):
        # theano variables
        X = T.ivector('X')
        Y = T.ivector('Y')
        LR = T.scalar('LR')
        SU, SV, SW = self.U, self.V, self.W
        # scan
        def forward(X, pr1, SU, SW, SV):
            S = T.tanh(SU[:, X] + SW.dot(pr1)) # wieght on input +wieght based on previous input
            dot = SV.dot(S)
            oSum = np.exp(dot - T.max(dot))
            o = oSum / T.sum(oSum) #output
            return [S, o]



        [sResults, oResults], Updates = theano.scan(fn=forward,
                                        outputs_info=[dict(initial=T.zeros(self.hiddenDim)), None],
                                        sequences=X,
                                        non_sequences=[SU, SW, SV],
                                        truncate_gradient=self.bpttTruncate,
                                        strict=True)

        #theano functions
        oError = T.sum(T.nnet.categorical_crossentropy(oResults, Y))
        dU = T.grad(oError, SU)
        dV = T.grad(oError, SV)
        dW = T.grad(oError, SW)

        self.s = theano.function([X], sResults)
        self.o = theano.function([X], oResults)
        self.predict = theano.function([X], T.argmax(oResults, axis=1))
        self.error = theano.function([X, Y], oError)
        self.bppt = theano.function([X, Y], [dU, dV, dW])
        self.sdgStep = theano.function([X, Y, LR], updates=[(self.U, self.U - LR * dU), (self.V, self.V - LR * dV), (self.W, self.W - LR * dW)])
    def calculateTotalLoss(self, X, Y):
        sum = 0

        for x, y in zip(X, Y):
            if len(x) != 0 and len(y) != 0:
                sum += self.error(x, y)
        return sum


    def calculateLoss(self, X, Y):
        numChar = np.sum([len(y) for y in Y])
        return self.calculateTotalLoss(X, Y) / float(numChar)

    def toFile(self, lst, file):

        # lst = self.U.get_value().tolist()
        with open(file, 'w') as myfile:
            json.dump(lst, myfile)

    def fromFile(self, file):
        with open(file, 'r') as infile:
            newList = json.load(infile)
            return np.asarray(newList)

