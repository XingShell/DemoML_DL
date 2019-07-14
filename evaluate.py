from sklearn import metrics
def evaluate(realLabel,predictionLabel):
    print('AUC: %.4f' % metrics.roc_auc_score(realLabel, predictionLabel))
    print('ACC: %.4f' % metrics.accuracy_score(realLabel, predictionLabel))
    print('Recall: %.4f' % metrics.recall_score(realLabel, predictionLabel))
    print('F1-score: %.4f' % metrics.f1_score(realLabel, predictionLabel))
    print('Precision: %.4f' % metrics.precision_score(realLabel, predictionLabel))
    print(metrics.confusion_matrix(realLabel, predictionLabel))