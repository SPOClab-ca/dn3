import sklearn.metrics as skmetrics


def _handle_cropped(y_p):
    """
    A straightforward helper that simply averages multiple crops if they are present.

    Parameters
    ----------
    y_p: np.ndarray
         The predicted values with shape batch x targets (x <optional crops>)

    Returns
    -------
    y_p_mean: np.ndarray
              If there is an additional crop dimensions, mean across this dimension
    """
    if len(y_p.shape) == 2:
        return y_p
    elif len(y_p.shape) == 3:
        return y_p.mean(-1)
    else:
        raise ValueError("Predictions should be 1 or 2 dimensions in shape (excluding batches)")


def _binarize_two_class(y_p):
    if y_p.shape[-1] == 2:
        return y_p[..., -1]
    elif y_p.shape[-1] > 2:
        print("This simple metric implementation doesn't support multi-class targets.")
        return 0


def auroc(inputs, outputs):
    y_p = _binarize_two_class(_handle_cropped(outputs.detach().cpu().numpy()))
    y_t = inputs[-1].detach().cpu().numpy()
    return skmetrics.roc_auc_score(y_t, y_p)


def balanced_accuracy(inputs, outputs):
    y_p = _binarize_two_class(_handle_cropped(outputs.detach().cpu().numpy()))
    y_t = inputs[-1].detach().cpu().numpy()
    return skmetrics.balanced_accuracy_score(y_t, y_p)
