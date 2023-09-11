import torch
from typing import Callable, List


class Metric:
    def __init__(self, name: str, precision: int = 2):
        self.name: str = name
        self.precision = precision

    def update(self, predictions, truth):
        raise NotImplementedError("Calling abstract base method of 'Metric' class")

    def reset(self):
        raise NotImplementedError("Calling abstract base method of 'Metric' class")

    def value(self):
        raise NotImplementedError("Calling abstract base method of 'Metric' class")


class MetricContainer:
    def __init__(self, metrics: List[Metric] = None):
        self.metrics: List[Metric] = metrics or []
        self.eval = False

    def set_train(self):
        # reset all metrics
        self.reset()
        self.eval = False

    def set_val(self):
        self.reset()
        self.eval = True

    def update(self, predictions, truth):
        for metric in self.metrics:
            metric.update(predictions, truth)

    def reset(self):
        for metric in self.metrics:
            metric.reset()

    def summary(self):
        metrics_logs = {}

        for metric in self.metrics:
            name = metric.name
            value = metric.value()

            if isinstance(value, dict):
                for (name, m) in value.items():
                    if self.eval:
                        if not name.startswith("val_"):
                            name = f"val_{name}"
                    metrics_logs[name] = m
            else:
                if self.eval:
                    if not name.startswith("val_"):
                        name = f"val_{name}"
                metrics_logs[name] = value

        return metrics_logs

    def summary_string(self):
        metrics_logs = {}

        for metric in self.metrics:
            name = metric.name
            value = metric.value()
            if isinstance(value, dict):
                for (name, m) in value.items():
                    if self.eval:
                        if not name.startswith("val_"):
                            name = f"val_{name}"
                    metrics_logs[name] = m, metric.precision
            else:
                if self.eval:
                    if not name.startswith("val_"):
                        name = f"val_{name}"
                metrics_logs[name] = value, metric.precision

        return "\t".join(
            [f'{{}}: {{:.{precision}f}}'.format(name, value) for name, (value, precision) in metrics_logs.items()])


class BinaryAccuracy(Metric):
    def __init__(self,
                 threshold=0.5,
                 name: str = "accuracy",
                 precision: int = 2
                 ):
        super(BinaryAccuracy, self).__init__(name, precision)
        self.threshold = threshold
        self.correct = 0
        self.total = 0

    def update(self, predictions, targets):
        predictions = torch.where(predictions >= self.threshold, 1, 0)
        correct = torch.where(predictions.eq(targets), 1, 0).sum().item()
        self.correct += correct
        self.total += len(predictions)

    def reset(self):
        self.correct = 0
        self.total = 0

    def value(self):
        return self.correct / self.total


class Accuracy(Metric):
    def __init__(self,
                 embeddings_to_metric: Callable = lambda x: x,
                 pos_min=0.5,
                 pos_max=1.0,
                 name: str = "accuracy",
                 precision: int = 2
                 ):
        super().__init__(name, precision)
        self.embeddings_to_metric = embeddings_to_metric
        self.correct = 0
        self.total = 0
        self.pos_min = pos_min
        self.pos_max = pos_max

    def update(self, predictions, truth):
        if isinstance(predictions, tuple):
            predictions = self.embeddings_to_metric(predictions)
        predictions = torch.where(
            torch.logical_and(
                predictions >= self.pos_min, predictions <= self.pos_max
            ),
            1,
            0
        )
        correct = torch.where(predictions.eq(truth), 1, 0).sum().item()
        self.correct += correct
        self.total += len(predictions)

    def reset(self):
        self.correct = 0
        self.total = 0

    def value(self):
        return self.correct / self.total


class ConfusionMatrix(Metric):
    def __init__(self,
                 embeddings_to_metric: Callable = lambda x: x,
                 pos_min=0.5,
                 pos_max=1.0,
                 precision: int = 2
                 ):
        super().__init__("ConfusionMatrix", precision)
        self.embeddings_to_metric = embeddings_to_metric
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
        self.pos_min = pos_min
        self.pos_max = pos_max

    def update(self, predictions, truth):
        if isinstance(predictions, tuple):
            predictions = self.embeddings_to_metric(predictions)
        predictions = torch.where(
            torch.logical_and(
                predictions >= self.pos_min, predictions <= self.pos_max
            ),
            1,
            0
        )
        tp = torch.where(torch.logical_and(
            truth == 1,
            predictions == 1
        ), 1, 0).sum().item()
        tn = torch.where(torch.logical_and(
            truth == 0,
            predictions == 0
        ), 1, 0).sum().item()
        fp = torch.where(torch.logical_and(
            truth == 0,
            predictions == 1
        ), 1, 0).sum().item()
        fn = torch.where(torch.logical_and(
            truth == 1,
            predictions == 0
        ), 1, 0).sum().item()

        self.tp += tp
        self.tn += tn
        self.fp += fp
        self.fn += fn

    def reset(self):
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0

    def value(self):
        return {
            "TP": self.tp,
            "TN": self.tn,
            "FP": self.fp,
            "FN": self.fn
        }




class Precision(Metric):
    def __init__(self,
                 threshold: float = 0.5,
                 name: str = "precision",
                 precision: int = 3,
                 epsilon=1e-10
                 ):
        super(Precision, self).__init__(
            name=name,
            precision=precision
        )
        self.threshold = threshold
        self.tp = 0
        self.fp = 0
        self.epsilon = epsilon

    def update(self, predictions, truth):
        predictions = torch.where(predictions >= self.threshold, 1, 0)
        self.tp += torch.where(
            torch.logical_and(
                predictions.eq(1),
                truth.eq(1)
            ),
            1, 0).sum().item()
        self.fp += torch.where(
            torch.logical_and(
                predictions.eq(1),
                truth.eq(0)
            ), 1, 0
        ).sum().item()

    def value(self):
        return self.tp / (self.tp + self.fp + self.epsilon)

    def reset(self):
        self.tp = 0
        self.fp = 0


class Recall(Metric):
    def __init__(self,
                 threshold: float = 0.5,
                 name: str = "recall",
                 precision: int = 3,
                 epsilon=1e-10
                 ):
        super(Recall, self).__init__(
            name=name,
            precision=precision
        )
        self.threshold = threshold
        self.tp = 0
        self.fn = 0
        self.epsilon = epsilon

    def update(self, predictions, truth):
        predictions = torch.where(predictions >= self.threshold, 1, 0)
        self.tp += torch.where(
            torch.logical_and(
                predictions.eq(1),
                truth.eq(1)
            ),
            1, 0).sum().item()
        self.fn += torch.where(
            torch.logical_and(
                predictions.eq(0),
                truth.eq(1)
            ), 1, 0
        ).sum().item()

    def value(self):
        return self.tp / (self.tp + self.fn + self.epsilon)

    def reset(self):
        self.tp = 0
        self.fn = 0
