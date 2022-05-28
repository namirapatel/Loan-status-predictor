import _pickle as cPickle
def loan_predict(data_input):
    with open('loan_predictor.pkl', 'rb') as fid:
        classifier = cPickle.load(fid)
    X_train_prediction = classifier.predict([data_input])
    return X_train_prediction[0]

print(loan_predict([1,0,0,1,0,5000,0,100,360,1,0]))
