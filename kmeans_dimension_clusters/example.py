import glob
import numpy as np
from kmeans import kmeans, avg_iou
import pdb

PROCESSED_LABEL_PATH = "/home/snk/WindowsDisk/Download/KITTI/labels/"
CLUSTERS = 5

def load_dataset(path):
	dataset = []
	for label_file in glob.glob("{}/*.txt".format(path)):
		label = np.loadtxt(label_file)
		if label.ndim == 1:
			label = np.expand_dims(label, 0)

		for obj in label:
			w = obj[3]
			h = obj[4]
			dataset.append([w, h])

	return np.array(dataset)

# (Pdb) data.shape
# (51865, 2)
data = load_dataset(PROCESSED_LABEL_PATH)
# pdb.set_trace()
out = kmeans(data, k=CLUSTERS)
print("Accuracy: {:.2f}%".format(avg_iou(data, out) * 100))
print("Boxes:\n {}".format(out))

ratios = np.around(out[:, 0] / out[:, 1], decimals=2).tolist()
print("Ratios:\n {}".format(sorted(ratios)))