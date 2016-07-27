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
        # self.U = theano.shared(name='U', value=U.astype(theano.config.floatX))
        # self.V = theano.shared(name='V', value=V.astype(theano.config.floatX))
        # self.W = theano.shared(name='W', value=W.astype(theano.config.floatX))



    def forwardProp(self, x):
        # python variables
        xLength = len(x)
        s = theano.shared(np.zeros((xLength + 1, self.hiddenDim)))
        o = np.zeros((xLength, self.wordDim))

        #theano variables
        K = T.iscalar("K")
        X = T.ivector("X")
        SU = T.dmatrix("SU")
        SW = T.dmatrix("SW")
        SV = T.dmatrix("SV")
        A = T.dvector("A")

        #theano functions
        def forwardS(prior_result, X, K, SU, SW):
            S = T.tanh(SU[:, X[K]] + T.dot(SW, prior_result))
            return S

        def forwardO(A, SV):
            dot = SV.dot(A)
            oSum = np.exp(dot - T.max(dot))
            o = oSum/T.sum(oSum)
            return o



        sResult, sUpdates = theano.scan(fn=forwardS,
                                        outputs_info=T.ones_like(s[K-1]),
                                        sequences=[],
                                        non_sequences=[X, K, SU, SW],
                                        n_steps=K)

        oResult, oUpdates = theano.scan(fn=forwardO,
                                        non_sequences=[A, SV],
                                        n_steps=K)
        finalS = sResult[-1]

        finalO = oResult[-1]

        calcS = theano.function(inputs=[X, K, SU, SW], outputs=finalS)

        calcO = theano.function(inputs=[A, K, SV], outputs=finalO)

        cS = calcS(x, xLength - 1, self.U, self.W)
        cO = calcO(cS, xLength -1, self.V)
        print(s.get_value().shape)
        print(cS.shape)
        t = np.zeros((xLength + 1, self.hiddenDim))
        for i in np.arange(xLength):
            t[i] = cS[i]
            o[i] = cO[i]
        print(t.shape)
        print(o.shape)

        # print(self.U[:, 0].shape)