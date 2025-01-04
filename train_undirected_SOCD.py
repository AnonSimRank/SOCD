import numpy as np
import scipy.sparse.linalg
from scipy import sparse
from collections import OrderedDict
import time

from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, normalized_mutual_info_score, accuracy_score

from Utils import SimpleNetwork2SpaMat_0, SimpleNetwork2SpaMat
from SOCD import SOCD
import warnings
warnings.filterwarnings('ignore')


# read data ##################################################################################
dataset_name = "football"
input_dir_name = r"input\{}".format(dataset_name)
output_dir_name = r"output\{}".format(dataset_name)

file_name = r"\{}.txt".format(dataset_name)
file_txt = input_dir_name + file_name

spa_mat = SimpleNetwork2SpaMat_0(file_txt, 0)
spa_mat_symmetric = spa_mat + spa_mat.transpose()
##############################################################################################

# row sum ###################################################################################
row_num = spa_mat_symmetric.shape[0]
dense_mat_row_sum = np.array(np.sum(spa_mat_symmetric, axis=1).tolist())
dense_mat_row_sum_1d = dense_mat_row_sum.reshape((row_num,))
#############################################################################################

row_num_arange = np.arange(row_num)
row_index_array = row_num_arange
col_index_array = row_num_arange

# diagonalize row sum #######################################################################
spa_diagonal_mat = sparse.csc_matrix((dense_mat_row_sum_1d, (row_index_array, col_index_array)), shape=(row_num, row_num))
#############################################################################################

# Laplacian matrix ###########################################################################
laplacian_mat = spa_diagonal_mat - spa_mat_symmetric
##############################################################################################

# initial vector 1 ######################################################################
#rng = np.random.default_rng(seed=42)
#init_vec = rng.random(row_num)
#print(init_vec)
#init_vec = rng.standard_normal(row_num)
#print(init_vec)
init_vec = np.ones(row_num)
init_vec[0] = 1 + row_num * np.sqrt(np.finfo(float).eps)
#######################################################################################
iter_num = 5 * int(np.ceil(np.sqrt(row_num)))
#iter_num = row_num
community_k = 12
# train ###############################################################################
t1 = time.time()
obj = SOCD(laplacian_mat, init_vec, community_k, iter_num)
#converge_Ritz_pair_dict, orth_mat_Q = obj.selective_orth_full()
k_means_ = obj.k_means()
t2 = time.time()
print("requires time:%s" % (t2 - t1))
train_labels = k_means_.labels_
############## check orthogonality ##################################

#orth_mat_Q_T_orth_mat_Q = np.matmul(orth_mat_Q.transpose(), orth_mat_Q)

# save train labels ##############################################################
train_labels_file_name = r"\{}_SOCD_train_labels_5.txt".format(dataset_name)
train_labels_file = output_dir_name + train_labels_file_name
np.savetxt(train_labels_file, train_labels, fmt='%d')
##########################################################################################

# evaluate ###############################################################################
true_labels_txt = r"\{}_true_labels.txt".format(dataset_name)
true_labels_file_txt = input_dir_name + true_labels_txt
true_labels_array = np.loadtxt(true_labels_file_txt, dtype=int)

#AMI_score = adjusted_mutual_info_score(true_labels_array, train_labels)
ARI_score = adjusted_rand_score(true_labels_array, train_labels)
NMI_score = normalized_mutual_info_score(true_labels_array, train_labels)
accuracy_score_ = accuracy_score(true_labels_array, train_labels)
############################################################################################

# save evaluate score ######################################################################
evaluation_dict = OrderedDict()
#evaluation_dict["AMI_score"] = AMI_score
evaluation_dict["ARI_score"] = ARI_score
evaluation_dict["NMI_score"] = NMI_score
evaluation_dict["accuracy_score"] = accuracy_score_

metric_score_txt = r"\{}_SOCD_metric_score_5.txt".format(dataset_name)
metric_score_file_txt = output_dir_name + metric_score_txt
with open(metric_score_file_txt, "w") as f:
    for k, v in evaluation_dict.items():
        f.write(str(k) + "\t" + str(v) + "\n")
###########################################################################################





