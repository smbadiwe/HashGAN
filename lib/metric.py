import numpy as np


class MAPs:
    def __init__(self, r):
        self.R = r

    @staticmethod
    def distance(a, b):
        return np.dot(a, b)

    def get_maps_by_feature(self, database, query):
        ips = np.dot(query.output, database.output.T)
        ids = np.argsort(-ips, 1)
        apx = []

        print("\nR: {}. ips: {}. ids: {}.".format(self.R, ips.shape, ids.shape))
        for i in range(ips.shape[0]):
            label = query.label[i, :].copy()
            if i == 0:
                print("label before reset:", label)
            label[label == 0] = -1
            db_label = database.label[ids[i, :][0: self.R], :]
            if i == 0:
                print("db_label: {}. label and db_label".format(db_label.shape))
                print(label)
                print(db_label)
            imatch = np.sum(db_label == label, 1) > 0
            if i == 0:
                print("\nimatch")
                print(np.sum(db_label == label, 1))
                print(imatch)
            rel = np.sum(imatch)
            px = np.cumsum(imatch).astype(float) / np.arange(1, self.R + 1, 1)
            if rel != 0:
                apx.append(np.sum(px * imatch) / rel)

        apx = np.array(apx)

        return np.mean(apx)
