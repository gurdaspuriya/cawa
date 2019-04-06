import numpy as np
from sklearn import metrics
import copy

def compute_f1(b_float, tlbl):

	micro = [];
	samples = [];
	macro = [];
	weighted = [];

	max_micro_f1 = 0;
	best_threshold = 0.01;

	for threshold in np.arange(0.01, 1.0, 0.01):
		b = b_float.copy();
		b[b < threshold] = 0;
		b[b > 0] = 1;
		f1 = metrics.f1_score(y_true=tlbl, y_pred=b, average='micro');
		if f1 > max_micro_f1:
			max_micro_f1 = f1;
			best_threshold = threshold;

	b = b_float;
	b[b < best_threshold] = 0;
	b[b > 0] = 1;


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

	return [best_threshold, micro, samples, macro, weighted]