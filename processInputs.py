import cPickle as pickle
import numpy as np
import json
import nltk

MAX_SENTENCE = 30
outputMask = []

def convertToLabels(labels, review):
    tokens = nltk.tokenize.word_tokenize(review)
    tokensToLabels = [labels[x.lower()] if x in labels else labels['<Unk>'] for x in tokens]
    if len(tokensToLabels) < MAX_SENTENCE:
        tokensToLabels =  [30003 for i in range(MAX_SENTENCE - len(tokensToLabels))]+ tokensToLabels+[labels['<EOR>']]
    elif len(tokensToLabels) >= MAX_SENTENCE:
        tokensToLabels = tokensToLabels[:MAX_SENTENCE] + [labels['<EOR>']]
    return tokensToLabels

def convertToOutput(labels, review):
    global outputMask
    tokens = nltk.tokenize.word_tokenize(review)
    tokensToLabels = [labels[x.lower()] if x in labels else labels['<Unk>'] for x in tokens]
    if len(tokensToLabels) < MAX_SENTENCE:
        tokensToLabels +=  [labels['<EOR>']]+[30003 for i in range(MAX_SENTENCE - len(tokensToLabels))]
        outputMask.append([1 if tokensToLabels[i] != 30001 else 0 for i in range(len(tokensToLabels))]+[0 for i in range(MAX_SENTENCE - len(tokensToLabels)+1)])
    elif len(tokensToLabels) >= MAX_SENTENCE:
        tokensToLabels = tokensToLabels[:MAX_SENTENCE] + [labels['<EOR>']]
        outputMask.append([1 if tokensToLabels[i] != 30001 else 0 for i in range(MAX_SENTENCE+1)])
    return tokensToLabels

def processInputs():
    global outputMask
    labelsDict = pickle.load(open('labelsDict'))
    reviews = json.load(open('../../opinion_abstracts/rottentomatoes.json'))
    #embeddings = np.load(open('embeddings'))
    inputs = []
    outputs = []
    count = 0
    print 'here'
    for movie in reviews:
        critics = movie['_critics']
        for review in critics:
            inputs.append( convertToLabels(labelsDict, critics[review]))
            outputs.append(convertToOutput(labelsDict, movie['_critic_consensus']))
            convertToOutput(labelsDict, movie['_critic_consensus'])

            count += 1
            if count %1000 == 0:
                print count

    np.save(open('X','w'),np.array(inputs))
    np.save(open('Y','w'),np.array(outputs))

    mask = np.array(outputMask)
    print mask.shape
    np.save(open('outputMask','w'), mask)

def info():
    reviews = json.load(open('../../opinion_abstracts/rottentomatoes.json'))
    lenList1 = []
    lenList2 = []
    lenList3 = []
    for movie in reviews:
        critics = movie['_critics']
        lenList3.append(len(critics))
        for review in critics:
            lenList1.append(len(nltk.tokenize.word_tokenize(critics[review])))
        lenList2.append((len(nltk.tokenize.word_tokenize(movie['_critic_consensus']))))
    print max(lenList1)
    print max(lenList2)
    print max(lenList3)
    print sum(lenList1)/len(lenList1)
    print sum(lenList2)/len(lenList2)
    print sum(lenList3)/len(lenList3)
    import matplotlib.pyplot as plt
    plt.hist(lenList1, 50)
    plt.show()
    plt.close()
    plt.figure()
    plt.hist(lenList2, 50)
    plt.show()
    plt.close()
    plt.figure()
    plt.hist(lenList3, 50)
    plt.show()



processInputs()
#info()
