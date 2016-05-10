# -*- coding: utf-8 -*-
"""
Created on Sat May  7 21:50:05 2016

@author: dhruv
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn import linear_model
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from time import time
from sklearn import metrics
from sklearn.utils.extmath import density
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier

def main():
    # Import the pandas package, then use the "read_csv" function to read
    # the labeled training data
    train = pd.read_csv("/Users/dhruv/Downloads/APT/InputData/train.tsv", header=0, delimiter="\t", quoting=3)

    # Initialize the "CountVectorizer" object, which is scikit-learn's
    # bag of words tool.
    vectorizer = CountVectorizer(analyzer = "word", \
                                 tokenizer = None, \
                                 preprocessor = None, \
                                 stop_words = None, \
                                 max_features = 5000)

    # fit_transform() does two functions: First, it fits the model
    # and learns the vocabulary; second, it transforms our training data
    # into feature vectors. The input to fit_transform should be a list of
    # strings.
    train_data_features = vectorizer.fit_transform(train.data)

    # Numpy arrays are easy to work with, so convert the result to an
    # array
    train_data_features = train_data_features.toarray()

    # Take a look at the words in the vocabulary
    vocab = vectorizer.get_feature_names()
    #print vocab

    """


    # Sum up the counts of each vocabulary word
    dist = np.sum(train_data_features, axis=0)

    # For each, print the vocabulary word and the number of times it
    # appears in the training set
    for tag, count in zip(vocab, dist):
        print count, tag

    print "Training the random forest..."
    from sklearn.ensemble import RandomForestClassifier

    # Initialize a Random Forest classifier with 100 trees
    forest = RandomForestClassifier(n_estimators = 100)

    # Fit the forest to the training set, using the bag of words as
    # features and the sentiment labels as the response variable
    #
    # This may take a few minutes to run
    forest = forest.fit( train_data_features, train["type"] )

    test = pd.read_csv("/Users/dhruv/Downloads/APT/Ninproject/test.csv", header=0, delimiter="\t", \
                       quoting=3 )

    # Verify that there are 25,000 rows and 2 columns
    #print test.shape

    # Get a bag of words for the test set, and convert to a numpy array
    test_data_features = vectorizer.transform(test.data)
    test_data_features = test_data_features.toarray()

    # Use the random forest to make sentiment label predictions
    result = forest.predict(test_data_features)

    print result


    from sklearn.naive_bayes import MultinomialNB
    clf_MNB = MultinomialNB().fit( train_data_features, train.type)

    predicted = clf_MNB.predict(test_data_features)
    print predicted """


    from sklearn.cross_validation import train_test_split
    training, testing = train_test_split(train, train_size = 0.8)

    training_data_features = vectorizer.fit_transform(training.data)
    training_data_features = training_data_features.toarray()

    testing_data_features = vectorizer.transform(testing.data)
    testing_data_features = testing_data_features.toarray()

    print("****** For Normal Frequency Classification ******")
    trainModel(training_data_features,training.type,testing_data_features, testing.type)

    #Adding TF_IDF

    from sklearn.feature_extraction.text import TfidfTransformer
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(training_data_features)

    # clf = MultinomialNB().fit(X_train_tfidf, training.type)

    X_test_tfidf = tfidf_transformer.transform(testing_data_features)

    #predicted = clf.predict(X_test_tfidf)

    print("****** For TF-IDF Classification ******")
    trainModel(X_train_tfidf,training.type,X_test_tfidf, testing.type)


    output = pd.DataFrame( data=testing )
    output.to_csv( "Bag_of_Words_model.csv", index=False, quoting=3 )

    ''' Using tdidf
    TRAINING_DATA = training_data_features
    TRAINING_CLASS = training.type
    TESTING_DATA = testing_data_features
    TESTING_CLASS = testing.type '''

    test_file = pd.read_csv("/Users/dhruv/Downloads/APT/InputData/random_test.tsv", header=0, delimiter="\t", quoting=3 )

    test_file_features = vectorizer.transform(test_file.data)
    test_file_features = test_file_features.toarray()

    print("****** For Random File ******")
    trainModel(X_train_tfidf,training.type,test_file_features, test_file.type)

    plot(X_train_tfidf,training.type,X_test_tfidf, testing.type)


def plot(TRAINING_DATA, TRAINING_CLASS,TESTING_DATA, TESTING_CLASS):
    """ START PLOT RELATED CODE """
    import matplotlib.pyplot as plt
    #plt.figure(figsize=(12, 8))
    #plt.title("Score")

    results = []
    for clf, name in (
            (MultinomialNB(), "Multinomial NB"),
            (KNeighborsClassifier(n_neighbors=10), "kNN"),
            (RandomForestClassifier(n_estimators=100), "Random Forest"),
            (linear_model.SGDClassifier(), "SVM")):
        print('=' * 80)
        print(name)
        results.append(benchmark(clf,TRAINING_DATA, TRAINING_CLASS,TESTING_DATA, TESTING_CLASS))

    indices = np.arange(len(results))

    results = [[x[i] for x in results] for i in range(4)]

    clf_names, score, training_time, test_time = results
    training_time = np.array(training_time) / np.max(training_time)
    test_time = np.array(test_time) / np.max(test_time)

    plt.figure(figsize=(12, 8))
    plt.title("Score")
    plt.barh(indices, score, .2, label="score", color='r')
    plt.barh(indices + .3, training_time, .2, label="training time", color='g')
    plt.barh(indices + .6, test_time, .2, label="test time", color='b')
    plt.yticks(())
    plt.legend(loc='best')
    plt.subplots_adjust(left=.25)
    plt.subplots_adjust(top=.95)
    plt.subplots_adjust(bottom=.05)

    for i, c in zip(indices, clf_names):
        plt.text(-.3, i, c)

    # Benchmark classifiers
    from time import time
    from sklearn.utils.extmath import density

def benchmark(clf, TRAINING_DATA, TRAINING_CLASS,TESTING_DATA, TESTING_CLASS):
    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(TRAINING_DATA, TRAINING_CLASS)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(TESTING_DATA)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    score = metrics.accuracy_score(TESTING_CLASS, pred)
    print("accuracy:   %0.3f" % score)

    if hasattr(clf, 'coef_'):
        print("dimensionality: %d" % clf.coef_.shape[1])
        print("density: %f" % density(clf.coef_))

    print()
    clf_descr = str(clf).split('(')[0]
    return clf_descr, score, train_time, test_time


""" END OF PLOT """


def trainModel(TRAINING_DATA, TRAINING_CLASS,TESTING_DATA, TESTING_CLASS ):
    clf_MNB = MultinomialNB().fit( TRAINING_DATA, TRAINING_CLASS)
    predicted = clf_MNB.predict(TESTING_DATA)

    print "Using Multinomial NB:"
    print "Accuracy: ", np.mean(predicted == TESTING_CLASS)


    print "Classification Report"
    print(metrics.classification_report(TESTING_CLASS, predicted))


    clfsvm = linear_model.SGDClassifier()
    clfsvm.fit(TRAINING_DATA, TRAINING_CLASS)
    predictedsvm = clfsvm.predict( TESTING_DATA )

    print "Using SVM :"
    print "Accuracy: ",np.mean(predictedsvm == TESTING_CLASS)

    print "Classification Report"
    print(metrics.classification_report(TESTING_CLASS, predictedsvm))

if __name__== '__main__':
    main()