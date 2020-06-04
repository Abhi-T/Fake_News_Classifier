import  pandas as pd
import re
import numpy as np
df=pd.read_csv('train.csv')
# print(df.head())

#get the independent features
X=df.drop('label',axis=1)
# print(X.head())

#get the dependent features
y=df['label']
# print(y.head())

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
df=df.dropna()
messages=df.copy()
messages.reset_index(inplace=True)
print(messages['title'][6])

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
corpus=[]
for i in range(0, len(messages)):
    review= re.sub('[^a-zA-Z]]', ' ', messages['title'][i])
    review=review.lower()
    review=review.split()

    review=[ps.stem(word) for word in review if not word in stopwords.words('english')]
    review=''.join(review)
    corpus.append(review)

#Applyting countvectorizer
#creating the bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=5000, ngram_range=(1,3))
X=cv.fit_transform(corpus).toarray()

# print(corpus)

y=messages['label']

#divide the dataset into train and test
from sklearn.model_selection import train_test_split
X_train,X_test,y_train, y_test=train_test_split(X,y, test_size=0.33, random_state=0)
# print(cv.get_feature_names()[:20])
# print(cv.get_params())

count_df=pd.DataFrame(X_train, columns=cv.get_feature_names())
# print(count_df.head())

import matplotlib.pyplot as plt
import itertools
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks=np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh=cm.max()/2.
    for i , j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j,i , cm[i,j], horizontalalignment="center", color="white" if cm[i,j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel("predicted label")
    plt.savefig("Metrics.png")
    plt.show()

#Multinomial Naive bias Algorithm
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
classifier=MultinomialNB()

classifier.fit(X_train, y_train)
pred=classifier.predict(X_test)
score=metrics.accuracy_score(y_test, pred)
print("accuracy: %0.3f" % score)
cm=metrics.confusion_matrix(y_test, pred)
plot_confusion_matrix(cm, classes=['FAKE','REAL'])

classifier.fit(X_train,y_train)
pred=classifier.predict(X_test)
score=metrics.accuracy_score(y_test,pred)
print(score)

