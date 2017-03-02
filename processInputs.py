import pickle
import numpy as np
import json
import nltk

def convertToLabels(labels, review):
    tokens = nltk.tokenize.word_tokenize(review)
    tokensToLabels = [labels[x.lower()] if x in labels else labels['<Unk>'] for x in tokens]
    #for x in labels:
#        if x.lower() in labels:
#            tokensToLabels.append(labels[x.lower()])
#        else:
#            tokensToLabels.append(labels['<Unk>'])
    return tokensToLabels

def processInputs():
    labelsDict = pickle.load(open('labelsDict'))
    reviews = json.load(open('../../opinion_abstracts/rottentomatoes.json'))
    inputs = []
    outputs = []
    count = 0
    print 'here'
    for movie in reviews:
        count += 1
        print count
        critics = movie['_critics']
        formatted_review = []
        for review in critics:
            formatted_review += convertToLabels(labelsDict, critics[review]) + [labelsDict['<EOR>']]
        inputs.append(formatted_review)
        outputs.append(convertToLabels(labelsDict, movie['_critic_consensus']))
    Lens = [len(x) for x in inputs]
    mx =  max(Lens)
    mean = sum(Lens)/len(Lens)
    inputMasks = []
    for i in range(len(inputs)):
        pad = [ 30003 for j in range(mx-len(inputs[i]))]
        mask =  [0 for j in range(mx-len(inputs[i]))]+[1 for j in range(len(inputs[i]))]
        inputMasks.append(mask)
        inputs[i] = pad + inputs[i]

    Lens = [len(x) for x in outputs]
    mx =  max(Lens)
    mean = sum(Lens)/len(Lens)
    outputMasks = []
    for i in range(len(outputs)):
        pad = [30003 for j in range(mx-len(outputs[i]))]
        mask = [1 for j in range(len(outputs[i]))] + [0 for j in range(mx-len(outputs[i]))]
        outputMasks.append(mask)
        outputs[i] = outputs[i] + pad

    np.save(open('labeled_inputs','w'), np.array(inputs))
    np.save(open('labeled_consesus','w'), np.array(outputs))
    np.save(open('outputMask','w'), np.array(outputMasks))
    np.save(open('inputMask','w'), np.array(inputMasks))
processInputs()
