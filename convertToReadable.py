import pickle
import numpy as np

MAX_TOKENS = 30000
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
        if count > MAX_TOKENS:
            break
        if count %100 == 0:
            print count
    mean = np.atleast_2d(np.mean(embeddings, axis = 0))
    embeddings = np.append(embeddings,mean, axis = 0)
    wordDict['<Unk>'] = count
    wordDict['<EOR>'] = count + 1
    embeddings = np.append(embeddings,np.random.rand(1,300), axis = 0)
    embeddings = np.append(embeddings, np.zeros((1,300)), axis = 0)
    print embeddings.shape
    pickle.dump(wordDict, open('labelsDict', 'w'))
    np.save(open('embeddings', 'w'), embeddings)
process()
