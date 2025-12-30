def format_multilabel_metrics(metrics, ignore_index: int = 0):
    metrics_reformatted = {}
    for key in metrics.keys():
        if metrics[key].numel() > 1:
            for i, val in enumerate(metrics[key]):
                if not i == ignore_index:
                    metrics_reformatted[key + "_" + str(i)] = val
        elif metrics[key].numel() == 1:  # Regr Scalar
            metrics_reformatted[key] = metrics[key].item()
    return metrics_reformatted
