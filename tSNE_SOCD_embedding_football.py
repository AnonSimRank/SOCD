from sklearn.manifold import TSNE
import plotly.express as px
import numpy as np
from scipy import sparse
from Utils import SimpleNetwork2SpaMat_0, SimpleNetwork2SpaMat
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, normalized_mutual_info_score, accuracy_score
import matplotlib.pyplot as plt
import pandas as pd

import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# read data ##################################################################################
dataset_name = "football"

file_txt = r"output\{0}\SOCD_embedding_matrix_5.txt".format(dataset_name)
labels_file_txt = r"input\{0}\{1}_true_labels.txt".format(dataset_name, dataset_name)

# read embedding matrix ########################################################
embedding_matrix = np.loadtxt(file_txt)
##############################################################################################

# read true labels ###########################################################################
true_labels = np.loadtxt(labels_file_txt, dtype=int)
##############################################################################################

feat_cols = ['pixel' + str(i+1) for i in range(embedding_matrix.shape[1])]
df = pd.DataFrame(embedding_matrix, columns=feat_cols)
df['label'] = true_labels
print("df::", df)
print("\n")

data_subset = df[feat_cols].values
print("data_subset::", data_subset)
print("data_subset type::", type(data_subset))

tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(data_subset)

df['x'] = tsne_results[:,0]
df['y'] = tsne_results[:,1]
print("df:::", df)
plt.figure(figsize=(6,4))
plt.xticks([])
plt.yticks([])
sns.scatterplot(
    x="x", y="y",
    hue="label",
    palette=sns.color_palette("hls", 12),
    data=df,
    legend=False,
    alpha=1,
    style="label"
)

save_dir = r"Fig\tSNE_SOCD_embedding_football.eps"
plt.savefig(save_dir, dpi=1200)
plt.show()
