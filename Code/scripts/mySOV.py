import numpy as np

def compute_sov(b, tlbl):
	sov = 0.0;
	sov_smooth = 0.0;
	accuracy = 0.0;
	overload_labels = [];
	overload_segments = [];
	for i in range(len(tlbl)):
		ground_segments = 1;
		pred_segments = 1;
		sov_temp = 0.0;
		sov_smooth_temp = 0.0;
		segment_dict = {};
		ground_truth = tlbl[i];
		predicted = b[i];
		overload_labels.append(float(len(set(predicted)))/len(set(ground_truth)));
		prev = ground_truth[0];
		start = 0;
		for j in range(1, len(ground_truth)):
			if ground_truth[j]==prev:
				continue;
			ground_segments += 1;
			if prev in segment_dict:
				segment_dict[prev].append([start, j-1]);
			else:
				segment_dict[prev] = [[start, j-1]];
			start = j;
			prev = ground_truth[j];

		if prev in segment_dict:
			segment_dict[prev].append([start, len(ground_truth)-1]);
		else:
			segment_dict[prev] = [[start, len(ground_truth)-1]];

		prev = predicted[0];
		start = 0;
		for j in range(1, len(predicted)):
			if predicted[j]==prev:
				continue;
			pred_segments += 1;
			if prev in segment_dict:
				for k in range(len(segment_dict[prev])):
					minov = float(max(min(j-1, segment_dict[prev][k][1]) - max(start, segment_dict[prev][k][0]) + 1, 0));
					maxov = float(max(j-1, segment_dict[prev][k][1]) - min(start, segment_dict[prev][k][0]) + 1);
					epsilon = maxov - minov;
					epsilon = min(epsilon, minov, int(0.5*(j-1-start+1)), int(0.5*(segment_dict[prev][k][1]-segment_dict[prev][k][0]+1)));
					sov_temp += (minov/maxov)*(segment_dict[prev][k][1] - segment_dict[prev][k][0] + 1);
					sov_smooth_temp += ((minov + epsilon)/maxov)*(segment_dict[prev][k][1] - segment_dict[prev][k][0] + 1);
			start = j;
			prev = predicted[j];
		if prev in segment_dict:
			for k in range(len(segment_dict[prev])):
				minov = float(max(min(len(predicted)-1, segment_dict[prev][k][1]) - max(start, segment_dict[prev][k][0]) + 1, 0));
				maxov = float(max(len(predicted)-1, segment_dict[prev][k][1]) - min(start, segment_dict[prev][k][0]) + 1);
				epsilon = maxov - minov;
				epsilon = min(epsilon, minov, int(0.5*(len(predicted)-1-start+1)), int(0.5*(segment_dict[prev][k][1]-segment_dict[prev][k][0]+1)));
				sov_temp += (minov/maxov)*(segment_dict[prev][k][1] - segment_dict[prev][k][0] + 1);
				sov_smooth_temp += ((minov + epsilon)/maxov)*(segment_dict[prev][k][1] - segment_dict[prev][k][0] + 1);

		overload_segments.append(float(pred_segments)/ground_segments);

		sov += sov_temp/len(ground_truth);
		sov_smooth += sov_smooth_temp/len(ground_truth);
		accuracy += float(sum([1 for j in range(len(ground_truth)) if ground_truth[j]==predicted[j]]))/len(ground_truth);
	return (sov/len(b), sov_smooth/len(b), accuracy/len(b), overload_labels, overload_segments);
