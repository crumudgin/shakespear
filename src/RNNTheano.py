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
        s = np.zeros((xLength + 1, self.hiddenDim))

        #theano variables
        K = T.iscalar("K")
        X = T.ivector("X")
        S = T.dmatrix("S")
        SS = T.dmatrix("SS")
        A = T.dvector("A")

        #theano functions
        def forward(X, K, S, SS):
            return np.tanh(S[X[K]] + SS.dot(S)) #this seems wrong, but whatever
            # return SS.dot(S)
        result, updates = theano.scan(fn=forward,
                                      sequences=[],
                                      non_sequences=[X,K,S, SS],
                                      n_steps=K)
        final_result = result[-1]

        power = theano.function(inputs=[X, K, S, SS], outputs=final_result)

        print(power(x, xLength-1, self.U, self.W))
        # for i in range(final_result):
        #     s[i] = final_result[i]

    def softmax(self,x):
        xt = np.exp(x - np.max(x))
        return xt / np.sum(xt)