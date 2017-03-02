#import keras
#import seq2seq
#from seq2seq.models import SimpleSeq2Seq
import numpy as np
import pickle


def convertData():
    labeledInputs = np.load(open('labeled_inputs'))
    embeddings = np.load(open('embeddings'))
    labelDict = pickle.load(open('labelsDict'))
    labeledOutput = np.load(open('labeled_consensus'))
    print labeledOutput.shape
    x_train = [np.concatenate([embeddings[x] for x in labeled], axis = 0) for labeled in labeledInputs]
    pickle.dump(x_train, open('x_train', 'w'))
    print 'ha'
    y_train = np.zeros((labeledOutput.shape[0],labeledOutput.shape[1], embeddings.shape[0]))
    count = 0
    for row in labeledInputs:
        rowInd = 0
        for x in row:
            y_train[count,rowInd,x] = 1
            rowInd += 1
            if count %100 ==0 :print count
        count += 1
    np.save(open('x_train','w'),x_train)
    np.save(open('y_train','w'),y_train)
#model = SimpleSeq2Seq(input_dim=5, hidden_dim=10, output_length=8, output_dim=8)
#model.compile(loss='mse', optimizer='rmsprop')
#model.fit(X_train, Y_train, nb_epoch=5, batch_size=32)
convertData()
