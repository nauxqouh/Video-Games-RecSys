
from sklearn.neighbors import KDTree, BallTree

class NearestNeighbors_fromscratch():
    def __init__(self, n_neighbors=5, algorithm='auto', metric='minkowski', p=2, leaf_size=30):
        """
        n_neighbors: number of neighbors to use, ~ K
        algorithm: used to compute the nn
            'kd_tree': use KDTree from sklearn
            'ball_tree': use BallTree from sklearn
            'brute': use a brute-force search
            'auto': attempt to decide the most appropriate algorithm based on the values passed to fit method.
        leaf_size: leaf size passed to BallTree or KDTree.
        metric: use for distance computation (['manhattan', 'euclidean', 'minkowski', 'chebyshev'])
        p: parameter for the Minkowski metric (p=1: manhattan, p=2: euclidean, p=p: minkowsk)
        """
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.metric = metric
        self.p = p
        self.X = None
        self.tree = None
        self._fit_method = None

    def fit(self, X):
        self.X = np.array(X)
        
        method = self.algorithm
        if self.algorithm == 'auto':
            method = self._auto_select_algorithm()

        self._fit_method = method
        if self._fit_method == 'kd_tree':
            self.tree = KDTree(self.X, metric=self.metric, leaf_size=self.leaf_size)
        elif self._fit_method == 'ball_tree':
            self.tree = BallTree(self.X, metric=self.metric, leaf_size=self.leaf_size)
        elif self._fit_method == 'brute':
            pass
        else:
            raise NotImplementedError(f"Algorithm {self._fit_method} is not supported.")

    def _auto_select_algorithm(self):
        """Select matching algorithm for 'auto' mode based on n_features of train dataset."""
        n_samples, n_features = self.X.shape

        if n_features > 50:
            return 'brute'
        elif n_features <= 20:
            return 'kd_tree'
        elif n_features <= 50 and n_samples >= 1000:
            return 'ball_tree'
        else:
            return 'brute'
            
    def kneighbors(self, X_new, return_distance=True):
        """Find K nearest neighbors with each points in X_new"""
        X_new = np.array(X_new)
        
        if self._fit_method == 'kd_tree':
            neigh_dist, neigh_ind = self.tree.query(X_new, k=self.n_neighbors)
        elif self._fit_method == 'ball_tree':
            neigh_dist, neigh_ind = self.tree.query(X_new, k=self.n_neighbors)
        elif self._fit_method == 'brute':
            dists = []
            for new_p in X_new:
                distances = [self._compute_distance(new_p, p_i) for p_i in self.X]
                dists.append(distances)
    
            dists = np.array(dists)
            neigh_ind = np.argsort(dists, axis=1)[:, :self.n_neighbors]
            neigh_dist = np.take_along_axis(dists, neigh_ind, axis=1)
        else:
            raise ValueError("Model not fitted properly.")

        if return_distance:
            return neigh_dist, neigh_ind
        else:
            return neigh_ind

    def _compute_distance(self, x, y):
        """Compute distance between each points in X_new and points in train dataset"""
        if self.metric == 'euclidean':
            return np.sqrt(np.sum((x - y) ** 2))
        elif self.metric == 'manhattan':
            return np.sum(np.abs(x - y))
        elif self.metric == 'minkowski':
            return np.sum(np.abs(x - y) ** self.p) ** (1 / self.p)
        elif self.metric == 'chebyshev':
            return np.max(np.abs(x - y))
        else:
            raise NotImplementedError(f"Metric {self.metric} is not supported.")

    def get_params(self):
        """Return model parameters in dictionary."""
        return {
            'n_neighbors': self.n_neighbors,
            'algorithm': self.algorithm,
            'leaf_size': self.leaf_size,
            'metric': self.metric,
            'p': self.p,
            'auto_selected_algorithm': self._fit_method
        }
