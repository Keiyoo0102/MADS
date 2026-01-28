import numpy as np

class Metric:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()

    def add_batch(self, predictions, labels):
        preds = predictions.flatten()
        targets = labels.flatten()
        mask = targets != 255
        preds = preds[mask]
        targets = targets[mask]
        x = preds + self.num_classes * targets
        bincount = np.bincount(x.astype(np.int64), minlength=self.num_classes ** 2)
        self.confusion_matrix += bincount.reshape(self.num_classes, self.num_classes)

    def evaluate(self):
        cm = self.confusion_matrix
        TP = np.diag(cm)
        FP = np.sum(cm, axis=0) - TP
        FN = np.sum(cm, axis=1) - TP
        EPS = 1e-10

        class_iou = TP / (TP + FP + FN + EPS)
        class_prec = TP / (TP + FP + EPS)
        class_recall = TP / (TP + FN + EPS)
        class_f1 = 2 * (class_prec * class_recall) / (class_prec + class_recall + EPS)

        miou = np.nanmean(class_iou)
        macc = np.nanmean(class_recall)
        mf1 = np.nanmean(class_f1)
        oa = np.sum(TP) / (np.sum(cm) + EPS)

        return {"OA": oa, "mIoU": miou, "mF1": mf1, "Class_IoU": class_iou}

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))