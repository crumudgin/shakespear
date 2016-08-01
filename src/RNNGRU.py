import numpy as np
import operator
import theano
import theano.tensor as T
import json

class RNNGRU:

    def __init__(self, wordDim, hiddenDim=100, bpttTruncate=4):
        self.wordDim = wordDim
        self.hiddenDim = hiddenDim
        self.bpttTruncate = bpttTruncate
        # B = np.zeros((3, hiddenDim))
        # C = np.zeros(wordDim)
        # E = np.random.uniform(-np.sqrt(1.0/wordDim), np.sqrt(1.0/wordDim), (hiddenDim,wordDim))
        # U = np.random.uniform(-np.sqrt(1.0/hiddenDim), np.sqrt(1.0/hiddenDim), (3, hiddenDim,hiddenDim))
        # V = np.random.uniform(-np.sqrt(1.0/hiddenDim), np.sqrt(1.0/hiddenDim), (wordDim, hiddenDim))
        # W = np.random.uniform(-np.sqrt(1.0/hiddenDim), np.sqrt(1.0/hiddenDim), (3, hiddenDim, hiddenDim))
        B = self.fromFile("B.txt")
        C = self.fromFile("C.txt")
        E = self.fromFile("E.txt")
        U = self.fromFile("U.txt")
        V = self.fromFile("V.txt")
        W = self.fromFile("W.txt")
        self.B = theano.shared(name='B', value=B.astype(theano.config.floatX))
        self.C = theano.shared(name='C', value=C.astype(theano.config.floatX))
        self.E = theano.shared(name='E', value=E.astype(theano.config.floatX))
        self.U = theano.shared(name='U', value=U.astype(theano.config.floatX))
        self.V = theano.shared(name='V', value=V.astype(theano.config.floatX))
        self.W = theano.shared(name='W', value=W.astype(theano.config.floatX))
        # self.mE = theano.shared(name='mE', value=np.zeros(E.shape).astype(theano.config.floatX))
        # self.mU = theano.shared(name='mU', value=np.zeros(U.shape).astype(theano.config.floatX))
        # self.mV = theano.shared(name='mV', value=np.zeros(V.shape).astype(theano.config.floatX))
        # self.mW = theano.shared(name='mW', value=np.zeros(W.shape).astype(theano.config.floatX))
        # self.mB = theano.shared(name='mb', value=np.zeros(B.shape).astype(theano.config.floatX))
        # self.mC = theano.shared(name='mc', value=np.zeros(C.shape).astype(theano.config.floatX))
        # self.toFile(self.B.get_value().tolist(), "B.txt")
        # self.toFile(self.C.get_value().tolist(), "C.txt")
        # self.toFile(self.E.get_value().tolist(), "E.txt")
        # self.toFile(self.U.get_value().tolist(), "U.txt")
        # self.toFile(self.V.get_value().tolist(), "V.txt")
        # self.toFile(self.W.get_value().tolist(), "W.txt")
        self.__theanoBuild__()



    def __theanoBuild__(self):
        # theano variables
        X = T.ivector('X')
        Y = T.ivector('Y')
        LR = T.scalar('LR')
        SE, SB, SC, SU, SV, SW = self.E, self.B, self.C, self.U, self.V, self.W
        decay = T.scalar('decay')
        # scan
        def forward(X, pr1):
            e = SE[:, X]
            z = T.nnet.hard_sigmoid(SU[0].dot(e) + SW[0].dot(pr1) + SB[0])
            r = T.nnet.hard_sigmoid(SU[1].dot(e) + SW[1].dot(pr1) + SB[1])
            c = T.tanh(SU[2].dot(e) + SW[2].dot(pr1 * r) + SB[2])
            s = (1 - z) * c+z*pr1
            # o = s
            o = T.nnet.softmax(SV.dot(s)+SC)[0]
            return [s, o]



        [sResults, oResults], Updates = theano.scan(fn=forward,
                                        outputs_info=[dict(initial=T.zeros(self.hiddenDim)), None],
                                        sequences=X,
                                        truncate_gradient=self.bpttTruncate)

        #theano functions
        oError = T.sum(T.nnet.categorical_crossentropy(oResults, Y))
        dB = T.grad(oError, SB)
        dC = T.grad(oError, SC)
        dE = T.grad(oError, SE)
        dU = T.grad(oError, SU)
        dV = T.grad(oError, SV)
        dW = T.grad(oError, SW)
        # mE = decay * self.mE + (1 - decay) * dE ** 2
        # mU = decay * self.mU + (1 - decay) * dU ** 2
        # mW = decay * self.mW + (1 - decay) * dW ** 2
        # mV = decay * self.mV + (1 - decay) * dV ** 2
        # mB = decay * self.mB + (1 - decay) * dB ** 2
        # mC = decay * self.mC + (1 - decay) * dC ** 2

        self.s = theano.function([X], sResults)
        self.o = theano.function([X], oResults)
        self.predict = theano.function([X], T.argmax(oResults, axis=1))
        self.error = theano.function([X, Y], oError)
        self.bppt = theano.function([X, Y], [dU, dV, dW, dE, dB, dC])
        self.sdgStep = theano.function([X, Y, LR], updates=[
                                                            # (self.mE, mE),
                                                            # (self.mB, mB),
                                                            # (self.mC, mC),
                                                            # (self.mU, mU),
                                                            # (self.mW, mW),
                                                            # (self.mV, mV),
                                                            (self.U, self.U - LR * dU),
                                                            (self.V, self.V - LR * dV),
                                                            (self.W, self.W - LR * dW),
                                                            (self.E, self.E - LR * dE),
                                                            (self.B, self.B - LR * dB),
                                                            (self.C, self.C - LR * dC)
                                                            ])
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

