import numpy as np
from sklearn import metrics

def compute_f1(b, tlbl):
	micro = [];
	samples = [];
	macro = [];
	weighted = [];

	micro.append(metrics.precision_score(y_true=tlbl, y_pred=b, average='micro'));
	micro.append(metrics.recall_score(y_true=tlbl, y_pred=b, average='micro'));
	micro.append(metrics.f1_score(y_true=tlbl, y_pred=b, average='micro'));
	
	samples.append(metrics.precision_score(y_true=tlbl, y_pred=b, average='samples'));
	samples.append(metrics.recall_score(y_true=tlbl, y_pred=b, average='samples'));
	samples.append(metrics.f1_score(y_true=tlbl, y_pred=b, average='samples'));
	
	macro.append(metrics.precision_score(y_true=tlbl, y_pred=b, average='macro'));
	macro.append(metrics.recall_score(y_true=tlbl, y_pred=b, average='macro'));
	macro.append(metrics.f1_score(y_true=tlbl, y_pred=b, average='macro'));
	
	weighted.append(metrics.precision_score(y_true=tlbl, y_pred=b, average='weighted'));
	weighted.append(metrics.recall_score(y_true=tlbl, y_pred=b, average='weighted'));
	weighted.append(metrics.f1_score(y_true=tlbl, y_pred=b, average='weighted'));

	return [micro, samples, macro, weighted]