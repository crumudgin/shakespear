import numpy as np
import operator
class RNNNumpy:

    def __init__(self, wordDim, hiddenDim=100, bpttTruncate=4):
        self.wordDim = wordDim
        self.hiddenDim = hiddenDim
        self.bpttTruncate = bpttTruncate
        self.U = np.random.uniform(-np.sqrt(1.0/wordDim), np.sqrt(1.0/wordDim), (hiddenDim,wordDim))
        self.V = np.random.uniform(-np.sqrt(1.0/hiddenDim), np.sqrt(1.0/hiddenDim), (wordDim, hiddenDim))
        self.W = np.random.uniform(-np.sqrt(1.0/hiddenDim), np.sqrt(1.0/hiddenDim), (hiddenDim, hiddenDim))

    def forwardPropagation(self, x):
        # print(self.U[:, 0].shape)
        # print(self.W.shape)
        # The total number of time steps
        T = len(x)
        # print(T)
        # During forward propagation we save all hidden states in s because need them later.
        # We add one additional element for the initial hidden, which we set to 0
        s = np.zeros((T + 1, self.hiddenDim))
        # The outputs at each time step. Again, we save them for later.
        o = np.zeros((T, self.wordDim))
        # For each time step...
        for t in np.arange(T):
            # print(t)
            # Note that we are indxing U by x[t]. This is the same as multiplying U with a one-hot vector.
            s[t] = np.tanh(self.U[:, x[t]] + self.W.dot(s[t - 1]))
            dot = self.V.dot(s[t])
            # print(self.V.dot(s[t]))
            o[t] = self.softmax(dot)
        print(s.shape)
        print(o.shape)
        return [o, s]

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def predict(self, x):
        # Perform forward propagation and return index of the highest score
        o, s = self.forwardPropagation(x)
        return np.argmax(o, axis=1)

    def calculateTotalLoss(self, x, y):
        L = 0
        for i in np.arange(len(y)):
            print(i)
            o, s = self.forwardPropagation(x[i])
            correctWordPredictions = o[np.arange(len(y[i])), y[i]]
            L += -1 * np.sum(np.log(correctWordPredictions))
        return L

    def calculateLoss(self, x, y):
        # print(y)
        N = np.sum((len(y_i) for y_i in y))
        return self.calculateTotalLoss(x,y)/N

    def bptt(self, x, y):
        T = len(y)
        o, s = self.forwardPropagation(x)
        dldU = np.zeros(self.U.shape)
        dldV = np.zeros(self.V.shape)
        dldW = np.zeros(self.W.shape)
        deltaO = o
        deltaO[np.arange(len(y)), y] -= 1
        for t in np.arange(T)[::-1]:
            dldV += np.outer(deltaO[t], s[t].T)
            deltaT = self.V.T.dot(deltaO[t]) * (1-(s[t] **2))
            for bpttStep in np.arange(max(0, t-self.bpttTruncate), t+1) [::-1]:
                dldW += np.outer(deltaT, s[bpttStep-1])
                dldU[:, x[bpttStep]] += deltaT
                deltaT = self.W.T.dot(deltaT) * (1- s[bpttStep-1] ** 2)
        return [dldU, dldV, dldW]

    def sdgStep(self, x, y, learningRate):
        dldU, dldV, dldW = self.bptt(x,y)
        self.U -= learningRate * dldU
        self.V -= learningRate * dldV
        self.W -= learningRate * dldW
        # print("sdgCompleet")
