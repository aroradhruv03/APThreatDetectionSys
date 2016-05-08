# -*- coding: utf-8 -*-
"""
Created on Sat May  7 21:50:05 2016

@author: dhruv
"""


categorie = ['attack','benign']

categories = ['alt.atheism']

"""from sklearn.datasets import fetch_20newsgroups
twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)

print twenty_train
target = open('/Users/dhruv/Downloads/APT/Ninproject/abc.json', 'a')
#target.write('Attack \n')
#word = word.encode('utf-8')
target.write(json.dumps(twenty_train))
#            target.write(' ')
target.close()"""
    


# Code for NB

"""import json
from pprint import pprint

with open('/Users/dhruv/Downloads/APT/Ninproject/packets.json') as data_file:    
    data = json.load(data_file)
pprint(data)


from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(data)"""





"""
# Code for SVM

from sklearn import svm
X = [[30, 0], [1, 1]]
y = [0, 1]
clf = svm.SVC()
fit = clf.fit(X, y)

print fit
print clf.predict([3,10])"""


# Import the pandas package, then use the "read_csv" function to read
# the labeled training data

import pandas as pd       
train = pd.read_csv("/Users/dhruv/Downloads/APT/Debo/train.tsv", header=0, delimiter="\t", quoting=3)


from sklearn.feature_extraction.text import CountVectorizer

# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool.  
vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None)#,   \
                             #,max_features = 5000) 

# fit_transform() does two functions: First, it fits the model
# and learns the vocabulary; second, it transforms our training data
# into feature vectors. The input to fit_transform should be a list of 
# strings.
train_data_features = vectorizer.fit_transform(train.data)



from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(train_data_features)




# Numpy arrays are easy to work with, so convert the result to an 
# array
train_data_features = train_data_features.toarray()

# Take a look at the words in the vocabulary
vocab = vectorizer.get_feature_names()
#print vocab

import numpy as np

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
print predicted