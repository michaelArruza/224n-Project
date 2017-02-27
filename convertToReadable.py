import pickle
import numpy as np

def process():
    f = open('../../glove.6B/glove.6B.300d.txt', 'r')
    line = f.readline().split()
    wordDict = {}
    wordDict[line[0]] = 0
    embeddings = np.atleast_2d(np.array([float(x) for x in line[1:]]))
    print embeddings.shape
    count = 1
    for line in f:
        line = line.split()
        wordDict[line[0]] = count
        count += 1
        embeddings = np.append(embeddings,np.atleast_2d(np.array([float(x) for x in line[1:]])), axis = 0)
    print embeddings.shape
    pickle.dump(wordDict, 'labelsDict')
    np.save('embeddings', embeddings)
process()
