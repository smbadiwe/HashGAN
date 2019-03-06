import numpy as np


class MAPs:
    def __init__(self, r):
        self.R = r

    @staticmethod
    def distance(a, b):
        return np.dot(a, b)

    def get_maps_by_feature(self, database, query):
        print("database:", database.output.shape, "query:", query.output.shape)
        ips = np.dot(query.output, database.output.T)
        ids = np.argsort(-ips, 1)
        print("ips:", ips.shape, "ids:", ids.shape)
        apx = []
        for i in range(ips.shape[0]):
            label = query.label[i, :].copy()
            # npz = np.nonzero(label == 0)[0]
            # if npz.shape[0] > 0:
            #     print("i:", i, "label:", label, "# apx:", len(apx), "npz:", npz.shape)
            label[label == 0] = -1
            imatch = np.sum(database.label[ids[i, :][0: self.R], :] == label, 1) > 0
            rel = np.sum(imatch)
            if rel != 0:
                # print("i:", i, "rel:", rel)
                px = np.cumsum(imatch).astype(float) / np.arange(1, self.R + 1, 1)
                apx.append(np.sum(px * imatch) / rel)

        return np.mean(np.array(apx))
