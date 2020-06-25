
import sklearn.metrics as skmetrics


def auroc(inputs, outputs):
    y_p = outputs.detach().cpu().numpy()
    y_t = inputs[-1].detach().cpu().numpy()
    if y_p.shape[-1] == 2:
        y_p = y_p[..., -1]
    elif y_p.shape[-1] > 2:
        print("This simple metric implementation doesn't support multi-class targets.")
        return 0
    return skmetrics.roc_auc_score(y_t, y_p)
