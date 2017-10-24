from sklearn.metrics import roc_auc_score, precision_score, recall_score, precision_recall_fscore_support
import numpy as np

def report(train_true, train_preds, test_true, test_preds, intents):
	
	print('Train AUC-ROC:', roc_auc_score(train_true, train_preds))
	print('Test AUC-ROC:', roc_auc_score(test_true, test_preds))

	train_preds = np.round(train_preds)
	test_preds = np.round(test_preds)

	print('TRAIN: (with repeats)')
	print("     type     precision     recall     f1-score     support")

	for ind, intent in enumerate(intents):
	    scores = np.asarray(precision_recall_fscore_support(train_true[:, ind], train_preds[:, ind]))[:, 1]
	    print("%s \t %f \t %f \t %f \t %f" % (intent, scores[0], scores[1], scores[2], scores[3]))

	print('TEST:')
	print("     type     precision     recall     f1-score     support")

	f1_scores = []

	for ind, intent in enumerate(intents):
	    scores = np.asarray(precision_recall_fscore_support(test_true[:, ind], test_preds[:, ind]))[:, 1]
	    print("%s \t %f \t %f \t %f \t %f" % (intent, scores[0], scores[1], scores[2], scores[3]))
	    f1_scores.append(scores[2])

	return(f1_scores)