class RNN:
    vocabulary = {}
    vocabularySize = len(vocabulary)

    def tokenizeSource(self, file):
        contents = open(file, 'r')
        counter = 1
        for i in contents:
            for j in i:
                if self.vocabulary.get(j, -1) == -1:
                    self.vocabulary[j] = counter
                    counter += 1


r = RNN()
r.tokenizeSource('works.txt')
print(r.vocabulary)
