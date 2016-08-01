import numpy as np
import RNNNumpy
import RNNTheano
import RNNGRU
import theano
import theano.tensor
import time
import random
import string





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
            hasPeriod = False
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
        preX = np.asarray([[np.int32(self.vocabulary[w]) for w in sent[:-1]] for sent in self.sentences])
        x = []
        for i in preX:
            if len(i) != 0:
                x.append(i)
        return x

    def yTrain(self):
        preY = np.asarray([[np.int32(self.vocabulary[w]) for w in sent[1:]] for sent in self.sentences])
        y = []
        for i in preY:
            if len(i) != 0:
                y.append(i)
        return y

    def train(self, model, xTrain, yTrain, evaluateLossAfter = 5, learningRate=.005, nepoch=100, save=100):
        losses = []
        numExamplesSeen = 0
        counter = 0
        for epoch in range(nepoch):
            print("nextRound")
            if(epoch % save == 0):
                G.toFile(G.U.get_value().tolist(),"U.txt")
                G.toFile(G.V.get_value().tolist(),"V.txt")
                G.toFile(G.W.get_value().tolist(),"W.txt")
                G.toFile(G.E.get_value().tolist(),"E.txt")
                G.toFile(G.B.get_value().tolist(),"B.txt")
                G.toFile(G.C.get_value().tolist(),"C.txt")
            if epoch % evaluateLossAfter == 0:
                loss = model.calculateLoss(xTrain, yTrain)
                losses.append((numExamplesSeen, loss))
                if len(losses) > 1 and losses[-1][1] > losses[-2][1]:
                    learningRate = learningRate * .5
            print("running sdg: "+str(counter))
            for i in range(len(yTrain)):
                model.sdgStep(xTrain[i], yTrain[i], learningRate)
                numExamplesSeen += 1
            counter += 1



    def generate(self, model, start):
        # print("start generation")
        newSentence = [self.vocabulary[start]]
        counter = 1
        while not newSentence[-1] == self.vocabulary['\n']:
            nextWordProbs = model.o(newSentence)
            sampledChar = -1
            # print(nextWordProbs[0])
            while sampledChar == -1 or sampledChar >=324:
                samples = np.random.multinomial(1, nextWordProbs[-1])
                sampledChar = np.argmax(samples)
            newSentence.append(sampledChar)
            if counter%1000 == 0:
                print([self.indexToChar[c] for c in newSentence])
            counter += 1
        # print("finished gen")
        # print([self.indexToChar[x] for x in newSentence])
        return "".join([self.indexToChar[x] for x in newSentence])


r = RNN()
r.tokenizeSource('input.txt')
# T = RNNTheano.RNNTheano(r.vocabularySize)
# r.train(T,r.xTrain(),r.yTrain(), nepoch=1000)
#
#
# T.toFile(T.U.get_value().tolist(),"U.txt")
# T.toFile(T.V.get_value().tolist(),"V.txt")
# T.toFile(T.W.get_value().tolist(),"W.txt")

G = RNNGRU.RNNGRU(r.vocabularySize)
r.train(G,r.xTrain(),r.yTrain(), nepoch=800)
# print(r.vocabulary)
# print(r.indexToChar)
i = 0
counter = 0
# print(string.ascii_uppercase)
while counter < 100:
    i = 0
    while i < 5:

        # print(counter)
        gen = r.generate(G, random.choice(string.ascii_uppercase))
        # print(gen)
        i = len(gen.split(" "))
    print(gen)
    counter += 1

# print(counter)




