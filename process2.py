import cPickle as pickle
import numpy as np
import json
import nltk

MAX_SENTENCE = 30
outputMask = []

def convertToLabels(labels, review):
    tokens = nltk.tokenize.word_tokenize(review)
    tokensToLabels = [labels[x.lower()] if x in labels else labels['<Unk>'] for x in tokens] + [labels['<EOR>']]
    if len(tokensToLabels) > MAX_SENTENCE:
        tokensToLabels = tokensToLabels[:MAX_SENTENCE] + [labels['<EOR>']]
    return tokensToLabels

def convertToOutput(labels, review):
    global outputMask
    tokens = nltk.tokenize.word_tokenize(review)
    tokensToLabels = [labels[x.lower()] if x in labels else labels['<Unk>'] for x in tokens]
    if len(tokensToLabels) < MAX_SENTENCE:
        tokensToLabels +=  [labels['<EOR>']]
        outputMask.append([1 if tokensToLabels[i] != 30001 else 0 for i in range(len(tokensToLabels))]+[0 for i in range(MAX_SENTENCE - len(tokensToLabels)+1)])
        tokensToLabels += [labels['<EOR>'] for i in range(MAX_SENTENCE - len(tokensToLabels)+1)]
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
        bigRev = []
        for review in critics:
            bigRev += convertToLabels(labelsDict, critics[review])
        if len(bigRev) < MAX_SENTENCE*20:
            bigRev =  [30003 for i in range(MAX_SENTENCE*20 - len(bigRev)+1)] + bigRev
        elif len(bigRev) >= MAX_SENTENCE*20:
            bigRev = bigRev[:MAX_SENTENCE*20] + [labelsDict['<EOR>']]
        inputs.append(bigRev)
        outputs.append(convertToOutput(labelsDict, movie['_critic_consensus']))
        count += 1
        if count %100 == 0:
            print count
    X2 = np.array(inputs)
    Y2 = np.array(outputs)
    print X2.shape
    print Y2.shape
    np.save(open('X2','w'),X2)
    np.save(open('Y2','w'),Y2)

    mask = np.array(outputMask)
    print mask.shape
    np.save(open('outputMask2','w'), mask)
processInputs()
