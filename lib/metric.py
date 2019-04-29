import numpy as np


class MAPs:
    def __init__(self, r):
        self.R = r

    @staticmethod
    def distance(a, b):
        return np.dot(a, b)

    def get_maps_by_feature(self, database, query, cfg):
        ips = np.dot(query.output, database.output.T)
        ids = np.argsort(-ips, 1)
        apx = []
        verbose = True
        if verbose:
            print("\nR: {}. ips: {}. ids: {}.".format(self.R, ips.shape, ids.shape))

        import matplotlib.pyplot as plt

        def read_image(filename):
            import os
            import cv2

            path = os.path.join(cfg.DATA.DATA_ROOT, filename)
            img = cv2.imread(path)
            wh = cfg.DATA.WIDTH_HEIGHT
            img = cv2.resize(img, (wh, wh), interpolation=cv2.INTER_AREA)
            return np.asarray(img)

        for i in range(ips.shape[0]):
            verbose = i == 0
            label = query.label[i, :].copy()
            if verbose:
                print("label before reset:", label)
            label[label == 0] = -1
            db_label = database.label[ids[i, :][0: self.R], :]
            if verbose:
                print("db_label: {}. label and db_label".format(db_label.shape))
                print(label)
                # print(db_label)
            imatch = np.sum(db_label == label, 1) > 0
            if verbose:
                print("\nimatch")
                print(np.sum(db_label == label, 1))
                print(imatch)
            rel = np.sum(imatch)
            px = np.cumsum(imatch).astype(float) / np.arange(1, self.R + 1, 1)
            if rel != 0:
                apx.append(np.sum(px * imatch) / rel)
                if verbose:
                    # get the image that was queried
                    qfile = query.input[i]
                    dfile = database.input[imatch]
                    print("rel:", rel, "- q file:", qfile, "- d_file:", dfile.shape)
                    plt.imshow(read_image(qfile))

        apx = np.array(apx)

        return np.mean(apx)
