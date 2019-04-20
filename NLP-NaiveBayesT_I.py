import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
nltk.download('stopwords')

dataset = pd.read_csv('../dm_files/twitter/train.csv', sep=',', engine='python')
testIMDB = pd.read_csv('../dm_files/imdb_data_3K.csv', sep=',', engine='python')

print(dataset.shape)

corpus = []
for i in range(0, 10000):
    text = re.sub('[^a-zA-Z]', ' ', dataset['SentimentText'][i])
    text = text.lower()
    text = text.split()
    ps = PorterStemmer()
    text = [ps.stem(word) for word in text if not word in set(stopwords.words('english'))]
    text = ' '.join(text)
    corpus.append(text)

corpusIMDB = []
for i in range(0, 3000):
    text = re.sub('[^a-zA-Z]', ' ', testIMDB['text'][i])
    text = text.lower()
    text = text.split()
    ps = PorterStemmer()
    text = [ps.stem(word) for word in text if not word in set(stopwords.words('english'))]
    text = ' '.join(text)
    corpusIMDB.append(text)

# creating Bag of words Model
cv = CountVectorizer(max_features=500)
X = cv.fit_transform(corpus).toarray()
Y = dataset.iloc[:, 1].values
Y = Y[0:10000]


X_imdb = cv.fit_transform(corpusIMDB).toarray()
Y_imdb = testIMDB.iloc[:, 0].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=21)

classifier = MultinomialNB()
classifier.fit(X_train, Y_train)



#predicting test values
# Y_pred =classifier.predict(X_test)
Y_Pred_imdb = classifier.predict(X_imdb)


con_matrix = confusion_matrix(Y_imdb, Y_Pred_imdb)
cr = classification_report(Y_imdb, Y_Pred_imdb)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Only use the labels that appear in the data
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


plot_confusion_matrix(con_matrix, classes=["0", "1"])

print(cr)

print('1')
#
#               precision    recall  f1-score   support
#
#            0       0.49      0.74      0.59      1500
#            1       0.48      0.24      0.32      1500
#
#    micro avg       0.49      0.49      0.49      3000
#    macro avg       0.49      0.49      0.46      3000
# weighted avg       0.49      0.49      0.46      3000
