disp_avlbl = True
import os
if 'DISPLAY' not in os.environ:
    disp_avlbl = False
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorly as tl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import scipy.sparse.linalg as lg
from time import time

from tensorly.decomposition import parafac
from sklearn.preprocessing import Normalizer

from .static_graph_embedding import StaticGraphEmbedding
from gem.utils import graph_util, plot_util
from gem.evaluation import visualize_embedding as viz

import sys
sys.path.append('./')
sys.path.append(os.path.realpath(__file__))

class TDNE(StaticGraphEmbedding):

    def __init__(self, *hyper_dict, **kwargs):
        '''
        Args:
            d: dimension of the embedding
            K: K-th order proximity
            R: rank of tensor decomposition
            n_iter: number of iteration in CP decomposition
        '''
        hyper_params = {
            'method_name': 'TDNE'
        }
        hyper_params.update(kwargs)
        for key in hyper_params.keys():
            self.__setattr__('_%s' % key, hyper_params[key])
        for dictionary in hyper_dict:
            for key in dictionary:
                self.__setattr__('_%s' % key, dictionary[key])

    def get_method_name(self):
        return self._method_name

    def get_method_summary(self):
        return '%s_%d' % (self._method_name, self._d)

    def learn_embedding(self, graph=None, edge_f=None,
                        is_weighted=False, no_python=False):
        if not graph and not edge_f:
            raise Exception('graph/edge_f needed')
        if not graph:
            graph = graph_util.loadGraphFromEdgeListTxt(edge_f)

        t1 = time()
        nNodes = graph.number_of_nodes()
        nEdges = graph.number_of_edges()
        print('num nodes: ', nNodes)
        print('num edges: ', nEdges)
        S = nx.to_numpy_matrix(graph, nodelist=sorted(graph.nodes()))
        A = Normalizer(norm='l1').fit_transform(S)

        if self._d==None:
            self._d = 2 * self._K
        else:
            assert self._d == 2 * self._K

        #Tensorization
        md_array = np.zeros((nNodes, nNodes, self._K))
        int_res = A
        for i in range(self._K):
            md_array[:, :, i] = int_res
            int_res = int_res.dot(A)
        XX = tl.tensor(md_array)
        print('Tensor shape: ', XX.shape)

        #CP Decomposition
        factors, errors = parafac(XX, rank=self._R, n_iter_max=self._n_iter, init='random', return_errors=True) #random_state=123,
        source_emb = factors[0]
        target_emb = factors[1]
        proximity_emb = factors[2]

        print('Source emb shape: ', source_emb.shape)
        print('Target emb shape: ', target_emb.shape)
        print('Proximity emb shape: ', proximity_emb.shape)

        source_proximity_emb = np.dot(source_emb, proximity_emb.T)
        target_proximity_emb = np.dot(target_emb, proximity_emb.T)
        emb = np.concatenate((source_proximity_emb, target_proximity_emb), axis=1)

        self._X = emb
        print("Embedding shape: ", self._X.shape)

        t2 = time()
        return self._X, (t2 - t1)

    def get_embedding(self):
        return self._X

    def get_edge_weight(self, i, j):
        return np.dot(self._X[i, :], self._X[j, :])

    def get_reconstructed_adj(self, X=None, node_l=None):
        if X is not None:
            node_num = X.shape[0]
            self._X = X
        else:
            node_num = self._node_num
        adj_mtx_r = np.zeros((node_num, node_num))
        for v_i in range(node_num):
            for v_j in range(node_num):
                if v_i == v_j:
                    continue
                adj_mtx_r[v_i, v_j] = self.get_edge_weight(v_i, v_j)
        return adj_mtx_r

if __name__ == '__main__':
    # load Zachary's Karate graph
    edge_f = 'data/karate.edgelist'
    G = graph_util.loadGraphFromEdgeListTxt(edge_f, directed=False)
    G = G.to_directed()
    res_pre = 'results/testKarate'
    graph_util.print_graph_stats(G)
    t1 = time()
    embedding = TDNE(d=None, K=6, R=2, n_iter=1000)
    embedding.learn_embedding(graph=G, edge_f=None, is_weighted=True, no_python=True)
    print('TDNE : \n\tTraining time: %f' % (time() - t1))
    viz.plot_embedding2D(embedding.get_embedding()[:, :2], di_graph=G, node_colors=None)
    plt.show()
