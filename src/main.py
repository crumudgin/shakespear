import numpy as np
from src import RNNNumpy

class RNN:
    vocabulary = {}
    vocabularySize = 0
    charByChar = []
    indexToChar = {}
    charToIndex = dict()
    sentences = []

    def tokenizeSource(self, file):
        contents = open(file, 'r')
        counter = 1
        for i in contents:
            sentence = []
            for j in i:
                if self.vocabulary.get(j, -1) == -1:
                    self.vocabulary[j] = counter
                    self.indexToChar[counter] = j
                    counter += 1
                self.charByChar.append(j)
                # self.indexToChar.append(self.vocabulary.get(j))
                if(j != '\n'):
                    sentence.append(j)
            self.sentences.append(sentence)
        self.vocabularySize = len(self.vocabulary)
        print(self.sentences)

    def xTrain(self):
        return np.asarray([[self.vocabulary[w] for w in sent[:-1]] for sent in self.sentences])

    def yTrain(self):
        return np.asarray([[self.vocabulary[w] for w in sent[1:]] for sent in self.sentences])


r = RNN()
r.tokenizeSource('works.txt')
print(r.vocabularySize)
print(r.charToIndex)
# print(r.vocabulary.values().)
print(r.xTrain()[0])
print(r.yTrain()[0])
n = RNNNumpy.RNNNumpy(r.vocabularySize)
# print(n.U)
# print(n.V)
# print(n.W)
print([r.indexToChar[i] for i in n.predict(r.xTrain()[0])])
