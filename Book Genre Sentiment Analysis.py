# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 18:33:07 2020

@author: carri
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 2019

@author: Daniel Mayo
"""

import numpy as np    # increases efficiency of matrix operations
import pandas as pd   # reads in data files of mixed data types
import re             # regular expressions to find/replace strings
import nltk           # natural language toolkit
from nltk.corpus import stopwords   # get list of stopwords to filter
                                    # out non-sentiment filler words
from sklearn.model_selection import train_test_split
import csv

csv.field_size_limit(191072)


stop_words = set(stopwords.words('english')) # make the stopword list a set
                                             # to increase speed of comparisons

df = pd.read_csv("Books.csv",engine = "python")

#print(df["Synopsis"])

# read the training data stored in "trainingDataXXXX.txt"
#test = pd.read_csv("testData.txt", header=0, delimiter="\t", quoting=3)     
# read the test data stored in testData.txt
# note: data files are tab delimited
  
X_train, X_test, y_train, y_test = train_test_split(df["Synopsis"], df["sci-fi and fantasy/ mystery and Suspense"], test_size=0.2)

#    
""" clean_my_text(): cleans the data with several replacements/deletions,
    tokenizes the text, and removes stopwords
    input: string data
    output: cleaned string data ready for sentiment analysis
"""
def clean_my_text(text):

    text = re.sub(r"\(([a-zA-Z])| .\)", " ", text) # takes out parenthesis
    
    
    text = re.sub(r"<.*?>", " ", text)      # quick removal of HTML tags
    text = re.sub("[^a-zA-Z]", " ", text)  # strip out all non-alpha chars
    text = re.sub("r'[^\x00-\x7F]+", " ", text)

    text = re.sub(" [A-Z][A-Z]+ "," ",text)
    
    
   
    
    text = text.strip().lower()            # convert all text to lowercase

    # taking out wierd leftovers from previous cleaning
    text = re.sub(" mon ", " ", text)
    text = re.sub(" gon ", " ", text)
    text = re.sub(" em ", " ", text)
    text = re.sub(" na ", " ", text)
    text = re.sub(" al ", " ", text)
    text = re.sub(" [a-z] ", " ", text) #taking out isolated letters by themself
    text = re.sub(" u ", " ", text) # there is an isolated u that always seems to appear
    text = re.sub(" \n\n"," ",text)
    text = re.sub(" \n", " ", text)
    text = re.sub(" re ", " ", text)
    text = re.sub(" don "," do not ", text)#changing don to do not
    text = re.sub(" doesn ", " does not ", text)#changing doesn to does not
    text = re.sub(" ll ", " ", text)
    text = re.sub(" ve ", " ", text)
    text = re.sub(" didn ", " did not ", text)# changning didn to did not
    text = re.sub(" ex ", " ",text)
    text = re.sub(" won ", " will not ",text) # changing won to will not
    text = re.sub(" [a-z] ", " ", text)  # taking out new isolated letters by themself

    tokenizer = nltk.tokenize.TreebankWordTokenizer()  # tokenizes text using
                                                       # smart divisions
    tokens = tokenizer.tokenize(text)      # store results in tokens
    

    unstopped = []                         # holds the cleaned data string
    for word in tokens:
        if word not in stop_words:         # removes stopwords
            unstopped.append(word)         # adds word to unstopped string
    stemmer = nltk.stem.WordNetLemmatizer()   # consolidates different
                                                # word forms
    cleanText = " ".join(stemmer.lemmatize(token) for token in unstopped)
                # joins final clean tokens into a string
    return cleanText



""" clean_my_data() calls clean_my_text for each line of text in a dataset
    category  
    input: data file containing raw text  
    output: data file containing cleaned text entries
"""
def clean_my_data(dataList):
    print("Cleaning all of the data")
    i = 0
    for textEntry in dataList:              # reads line of text under 
                                                    # review category
        cleanElement = clean_my_text(textEntry)     # cleans line of text
        dataList[i] = cleanElement   # stores cleaned text
        i = i + 1
        if (i % 5 == 0):
            print("Cleaning review number", i, "out of", len(dataList))
    print("Finished cleaning all of the data\n")
    return dataList


print("Operating on training data...\n")
reviews = X_train.tolist()
cleanReviewData = clean_my_data(reviews)            # cleans the training data

#==================================================================================================

#print (cleanReviewData[0])

#==================================================================================================

""" create_bag_of_words() generates the bag of words used to evaluate sentiment
    input: cleaned dataset
    output: tf-idf weighted sparse matrix
"""
def create_bag_of_words(X):
    from sklearn.feature_extraction.text import CountVectorizer
        # use scikit-learn for vectorization
    
    print ('Generating bag of words...')
    
    vectorizer = CountVectorizer(analyzer = "word",   \
                                 tokenizer = None,    \
                                 preprocessor = None, \
                                 stop_words = None,   \
                                 ngram_range = (1,2), \
                                 max_features = 10000)
        # generates vectorization for ngrams of up to 2 words in length
        # this will greatly increase feature size, but gives more accurate
        # sentiment analysis since some word combinations have large
        # impact on sentiment ie: ("not good", "very fast")
                                                         
    train_data_features = vectorizer.fit_transform(X)
        # vectorizes sparse matrix
    train_data_features = train_data_features.toarray()
        # convert to a NumPy array for efficient matrix operations
    from sklearn.feature_extraction.text import TfidfTransformer
    tfidf = TfidfTransformer()
    tfidf_features = tfidf.fit_transform(train_data_features)
        # use tf-idf to weight features - places highest sentiment value on
        # low-frequency ngrams that are not too uncommon 
    return vectorizer, tfidf_features, tfidf



vectorizer, tfidf_features, tfidf  = (create_bag_of_words(cleanReviewData))   
        # stores the sparse matrix of the tf-idf weighted features


""" train_logistic_regression() uses logistic regression model to
    evaluate sentiment
    options: C sets how strong regularization will be: large C = small amount
    input: tf-idf matrix and the sentiment attached to the training example
    output: the trained logistic regression model
"""
def train_logistic_regression(features, label):
    print ("Training the logistic regression model...")
    from sklearn.linear_model import LogisticRegression
    ml_model = LogisticRegression(C = 100, random_state = 0, solver = 'liblinear')
    ml_model.fit(features, label)
    print ('Finished training the model\n')
    return ml_model


ml_model = train_logistic_regression(tfidf_features, y_train)
    # holds the trained model
    
print("Operating on test data...\n")
sentiments = X_test.tolist()
cleanTestData = clean_my_data(sentiments)
    # cleans the test data for accuracy evaluation

test_data_features = vectorizer.transform(cleanTestData)
test_data_features = test_data_features.toarray()
    # vectorizes the test data

test_data_tfidf_features = tfidf.fit_transform(test_data_features)
test_data_tfidf_features = test_data_tfidf_features.toarray()
    # tf-idf of test data ngrams

predicted_y = ml_model.predict(test_data_tfidf_features)
    # uses the trained logistic regression model to assign sentiment to each
    # test data example

correctly_identified_y = predicted_y == y_test
accuracy = np.mean(correctly_identified_y) * 100
print ('The accuracy of the model in predicting movie review sentiment is %.0f%%' %accuracy)
    # compares the predicted sentiment (predicted_y) vs the actual 
# value stored in "sentiment"