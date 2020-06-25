import sklearn.metrics as skmetrics


def auroc(inputs, outputs):
    y_p = outputs.numpy()
    y_t = inputs[-1].numpy()
    return skmetrics.roc_auc_score(y_t, y_p)
