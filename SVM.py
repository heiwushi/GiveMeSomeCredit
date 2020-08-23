from sklearn import svm



def call(train_input, train_label,validate_input, validate_label):
    print("call")
    clf = svm.SVC(probability=True)
    clf.fit(train_input, train_label)
    return clf.predict(validate_input)