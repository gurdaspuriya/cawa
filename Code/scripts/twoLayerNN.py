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

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
torch.manual_seed(1194)

class DocClassifier(nn.Module):

	def __init__(self, hidden_dim, label_size, vocab_size):
		super(DocClassifier, self).__init__()
		self.dropout_utility = nn.Dropout(0.50);
		self.hidden_dim = hidden_dim
		self.vocab_size = vocab_size
		self.label_size = label_size

		self.input2hidden = nn.Linear(vocab_size, hidden_dim)
		self.hidden2label = nn.Linear(hidden_dim, label_size)


	def forward(self, docs, training=False):

		# Fully connected network
		x = F.relu(self.input2hidden(docs)); 
		if training:
			x = self.dropout_utility(x);
		y = self.hidden2label(x)
		log_probs = F.log_softmax(y, dim=1)
		return log_probs;

def classifier(training_data, learning_rate=0.1, num_epoch=10, hidden_dim=50, model=None, optimizer=None, tfidf_transformer=None):
	train_text = [];
	train_labels = [];
	
	for topic in range(len(training_data)):
		train_labels.extend([topic]*len(training_data[topic]));
		for i in range(len(training_data[topic])):
			train_text.append(training_data[topic][i]);
	if tfidf_transformer is None:
		tfidf_transformer = TfidfVectorizer(use_idf=False, binary=False, min_df = 0, norm=None, token_pattern='\S+').fit(train_text)
	vocab_size = len(tfidf_transformer.get_feature_names());
	X_train = tfidf_transformer.transform(train_text);
	X_matrix = autograd.Variable(torch.from_numpy(X_train.toarray()).float());
	X_labels = autograd.Variable(torch.LongTensor(train_labels));

	if model is None:
		model = DocClassifier(hidden_dim=hidden_dim, label_size=len(training_data), vocab_size=vocab_size);
	loss_function = nn.NLLLoss()
	if optimizer is None:
		optimizer = optim.Adam(model.parameters(),lr=learning_rate);
	model.train();
	for epoch in range(num_epoch):
		model.zero_grad()
		#Run our forward pass.
		log_probs = model(X_matrix, training=True)
		#Compute the loss, gradients, and update the parameters by calling optimizer.step()
		loss = loss_function(log_probs, X_labels)
		loss.backward()
		optimizer.step()
	return [tfidf_transformer, model, optimizer];

def test_performance(test_data, clf):
	test_text = [];
	test_labels = [];

	for topic in range(len(test_data)):
		test_labels.extend([topic]*len(test_data[topic]));
		for i in range(len(test_data[topic])):
			test_text.append(test_data[topic][i]);
	vocab_size = len(clf[0].get_feature_names());
	X_test = clf[0].transform(test_text)

	X_matrix = autograd.Variable(torch.from_numpy(X_test.toarray()).float());
	X_labels = autograd.Variable(torch.LongTensor(test_labels));
	log_probs = clf[1](X_matrix);
	predicted = log_probs.data.max(1)[1]  # get the index of the max log-probability

	return accuracy_score(test_labels, predicted)

def get_loss(test_text, clf, label_set):
	X_test = clf[0].transform(test_text);
	X_matrix = autograd.Variable(torch.from_numpy(X_test.toarray()).float());
	log_probs = np.exp(clf[1](X_matrix).data);

	scores = [[0 for i in range(len(label_set))] for j in range(len(log_probs))];
	for i in range(len(log_probs)):
		for j in range(len(label_set)):
			scores[i][j] = log_probs[i,label_set[j]];
	return scores;

def get_loss_simple(test_text, clf, label_set):
	X_test = clf[0].transform(test_text);
	X_matrix = autograd.Variable(torch.from_numpy(X_test.toarray()).float());
	log_probs = np.exp(clf[1](X_matrix).data);

	scores1 = [-1*float("inf")]*len(log_probs);
	preds1 = [""]*len(log_probs);
	scores2 = [-1*float("inf")]*len(log_probs);
	preds2 = [""]*len(log_probs);
	for i in range(len(log_probs)):
		for j in range(len(log_probs[i])):
			if j in label_set:
				if log_probs[i,j] > scores1[i]:
					scores2[i] = scores1[i];
					preds2[i] = preds1[i];
					scores1[i] = log_probs[i,j];
					preds1[i] = j;
				elif log_probs[i,j] > scores2[i]:
					scores2[i] = log_probs[i,j];
					preds2[i] = j;
	return [preds1, scores1, preds2, scores2];