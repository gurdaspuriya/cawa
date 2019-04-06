import numpy as np
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

def classifier(training_data):
    train_text = [];
    train_labels = [];
    
    for topic in range(len(training_data)):
        train_labels.extend([topic]*len(training_data[topic]));
        for i in range(len(training_data[topic])):
            train_text.append(training_data[topic][i]);
    tfidf_transformer = TfidfVectorizer(use_idf=False, binary=False, min_df = 0, norm=None, token_pattern='\S+').fit(train_text)
    X_train = tfidf_transformer.fit_transform(train_text);
    return [tfidf_transformer, LogisticRegression(random_state=12).fit(X_train, train_labels)];

def test_performance(test_data, clf):
    test_text = [];
    test_labels = [];

    for topic in range(len(test_data)):
        test_labels.extend([topic]*len(test_data[topic]));
        for i in range(len(test_data[topic])):
            test_text.append(test_data[topic][i]);
            
    X_test = clf[0].transform(test_text)
    predicted = clf[1].predict(X_test)
    return accuracy_score(test_labels, predicted)

def get_loss(test_text, clf, label_set):
    X_test = clf[0].transform(test_text);
    classes = clf[1].classes_.tolist();
    label_index = [];
    for label in label_set:
        label_index.append(classes.index(label));
    decision = 1. / (1 + np.exp(-1*clf[1].decision_function(X_test)));
    scores = [[0 for i in range(len(label_set))] for j in range(len(decision))];
    for i in range(len(decision)):
        for j in range(len(label_index)):
            scores[i][j] = decision[i][label_index[j]];
    return scores;

def get_loss_simple(test_text, clf, label_set):
    X_test = clf[0].transform(test_text);
    classes = clf[1].classes_;
    decision = 1. / (1 + np.exp(-1*clf[1].decision_function(X_test)));
    scores1 = [-1*float("inf")]*len(decision);
    preds1 = [""]*len(decision);
    scores2 = [-1*float("inf")]*len(decision);
    preds2 = [""]*len(decision);
    for i in range(len(decision)):
        for j in range(len(classes)):
            if classes[j] in label_set:
                if decision[i][j] > scores1[i]:
                    scores2[i] = scores1[i];
                    preds2[i] = preds1[i];
                    scores1[i] = decision[i][j];
                    preds1[i] = classes[j];
                elif decision[i][j] > scores2[i]:
                    scores2[i] = decision[i][j];
                    preds2[i] = classes[j];
    return [preds1, scores1, preds2, scores2];