import numpy as np
import scipy
from collections import OrderedDict, defaultdict
np.set_printoptions(precision=16)
import sklearn
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_mutual_info_score, adjusted_rand_score

class SOCD_Demmel:

    def __init__(self, her_mat, init_vec, community_k, iter_num, tol=np.finfo(float).eps):

        self.her_mat = her_mat
        self.init_vec = init_vec
        self.community_k = community_k
        self.tol = tol
        self.iter_num = iter_num

    def check_normalize(self):

        init_vec_norm = np.linalg.norm(self.init_vec)
        if int(init_vec_norm ** 2) == 1:
            normalize_init_vec = self.init_vec
        else:
            normalize_init_vec = self.init_vec / init_vec_norm

        return normalize_init_vec


    def selective_orth(self):

        row_num = self.her_mat.shape[0]

        orth_mat_Q = np.zeros((row_num, self.iter_num + 1))
        orth_mat_Q[:,1] = self.check_normalize()

        vec_alpha = np.zeros(self.iter_num)
        vec_beta = np.zeros(self.iter_num + 1)
        for j in range(1, self.iter_num):

            # Lanczos process ###################################
            r = self.her_mat * orth_mat_Q[:,j] - vec_beta[j-1] * orth_mat_Q[:,j-1]
            alpha_j = np.matmul(orth_mat_Q[:,j].transpose(), r)
            vec_alpha[j-1] = alpha_j
            r = r - alpha_j * orth_mat_Q[:,j]
            beta_j = np.linalg.norm(r)
            ######################################################################

            Q_j = orth_mat_Q[:,1:j+1]

            # eigen decomposition (Relatively Robust Representations) #############
            diag_ele = vec_alpha[0:j]
            off_diag_ele = vec_beta[1:j]
            [eigen_values, eigen_vectors] = scipy.linalg.eigh_tridiagonal(diag_ele, off_diag_ele, eigvals_only=False, select='a', lapack_driver='auto')
            #######################################################################

            # judge converge eigen pair ###########################################
            beta_last_entry_eigen_vec = beta_j * np.abs(eigen_vectors[-1])
            max_single_value = np.max(np.abs(eigen_values))
            converge_indexes = np.where(beta_last_entry_eigen_vec <= max_single_value * np.sqrt(self.tol))[0]
            converge_indexes_size = converge_indexes.size
            #######################################################################

            if converge_indexes_size != 0:
                for i in range(converge_indexes_size):
                    # compute and save converge eigen pair ################################
                    converge_pair_index = converge_indexes[i]
                    converge_eigen_val = eigen_values[converge_pair_index]
                    converge_eigen_vec = np.matmul(Q_j, eigen_vectors[:,converge_pair_index])

                    ########################################################################

                    # Lanczos vector is orthogonalized against converge Ritz vector##########
                    Lanczos_vec_Ritz_vec_dot_prod = np.matmul(r.transpose(), converge_eigen_vec)
                    r = r - Lanczos_vec_Ritz_vec_dot_prod * converge_eigen_vec
                    #########################################################################

            beta_j = np.linalg.norm(r)
            if beta_j < self.tol or j == self.iter_num - 1:
                orth_mat_Q_ = orth_mat_Q[:, 1:j + 1]
                vec_alpha_ = vec_alpha[0:j]
                vec_beta_ = vec_beta[1:j]
                [eigen_values, eigen_vectors] = scipy.linalg.eigh_tridiagonal(vec_alpha_, vec_beta_, eigvals_only=False, select='a', lapack_driver='auto')
                eigen_vectors_ = np.matmul(orth_mat_Q_[:, 0:j], eigen_vectors)
                return eigen_values, eigen_vectors_, orth_mat_Q_
            else:
                vec_beta[j] = beta_j
                orth_mat_Q[:,j+1] = r / beta_j

    def k_means(self):

        # intercept the first community_k items of the dictionary, that is, #########################
        # the smallest community_k Ritz values corresponding to the Ritz vectors ####################
        eigen_values, eigen_vectors, orth_Q = self.selective_orth()
        orth_Q_T_orth_Q = np.matmul(orth_Q.transpose(), orth_Q)
        eigen_vectors_k = eigen_vectors[:,0:self.community_k]
        Ritz_pair_values_array = np.array(eigen_vectors_k)
        ############################################################################################

        # the eigenvectors are orthogonal to each other corresponding to the different eigenvalues
        #orth_Ritz_pair_vectors, upper_tri_Ritz_pair_vectors = scipy.linalg.qr(Ritz_pair_values_array, mode='economic')
        ###########################################################################################

        # k-means algorithm ###################
        k_means_ = KMeans(n_clusters=self.community_k,  init='k-means++', random_state=0, n_init=10).fit(Ritz_pair_values_array)
        #######################################
        return k_means_










