import numpy as np
import RNNNumpy
import RNNTheano
import theano
import theano.tensor





class RNN:
    vocabulary = {}
    vocabularySize = 0
    charByChar = []
    indexToChar = {}
    charToIndex = dict()
    sentences = []

    def tokenizeSource(self, file):
        contents = open(file, 'r')
        counter = 0
        for i in contents:
            sentence = []
            # print(i)
            for j in i:
                # print(j)
                if self.vocabulary.get(j, -1) == -1:
                    self.vocabulary[j] = counter
                    self.indexToChar[counter] = j
                    counter += 1
                self.charByChar.append(j)
                # self.indexToChar.append(self.vocabulary.get(j))
                sentence.append(j)
            self.sentences.append(sentence)
        self.vocabulary['|'] = counter
        self.vocabularySize = len(self.vocabulary)
        # print(self.sentences)

    def xTrain(self):
        x = np.asarray([[np.int32(self.vocabulary[w]) for w in sent[:-1]] for sent in self.sentences])
        return x

    def yTrain(self):
        return np.asarray([[self.vocabulary[w] for w in sent[1:]] for sent in self.sentences])

    def train(self, model, xTrain, yTrain, evaluateLossAfter, learningRate=.005, nepoch=100):
        losses = []
        numExamplesSeen = 0
        counter = 0.0
        for epoch in range(nepoch):
            print("nextRound")
            if epoch % evaluateLossAfter == 0:
                loss = model.calculateLoss(xTrain, yTrain)
                losses.append((numExamplesSeen, loss))
                if len(losses) > 1 and losses[-1][1] > losses[-2][1]:
                    learningRate = learningRate * .5
            print("running sdg: "+str((counter/nepoch)*10)+"%")
            for i in range(len(yTrain)):
                model.sdgStep(xTrain[i], yTrain[i], learningRate)
                numExamplesSeen += 1
            counter += 1

    def generate(self, model):
        print("start generation")
        newSentence = [self.vocabulary['a']]
        counter = 0
        while not newSentence[-1] == self.vocabulary['\n']:
            nextWordProbs = model.forwardPropagation(newSentence)
            sampledChar = -1
            # print(nextWordProbs[0])
            while sampledChar == -1 or sampledChar >=324:
                samples = np.random.multinomial(1, nextWordProbs[0][0])
                sampledChar = np.argmax(samples)
            newSentence.append(sampledChar)
            # print(nextWordProbs[0])
            max = 0
            maxval = 0
            # for i in range(0,len(nextWordProbs[0][0])):
            #     if nextWordProbs[0][0][i] > maxval :
            #         max = nextWordProbs[0][0][i]
            #         maxval = i
            # # samples = np.random.multinomial(1, nextWordProbs[0])
            # # sampledWord = np.argmax(samples)
            # newSentence.append(maxval)
            if counter%1000 == 0:
                print([self.indexToChar[c] for c in newSentence])
            counter += 1
        print("finished gen")
        return [self.indexToChar[x] for x in newSentence[1:]]


r = RNN()
r.tokenizeSource('output.txt')
n = RNNNumpy.RNNNumpy(r.vocabularySize)
n.forwardPropagation(r.xTrain()[100])
print("--------------------------------------")
#
T = RNNTheano.RNNTheano(r.vocabularySize)
T.forwardProp(r.xTrain()[100])

# print(r.generate(n))

