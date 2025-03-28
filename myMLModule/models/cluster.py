import numpy as np

class KMeans:
    """
        KMeans聚类

        属性:
            n_clusters: 簇数量
            n_epoches: 迭代次数
            tolerance: 中心点偏移量阈值

    """
    def __init__(self, n_clusters, n_epoches=200, tolerance=1e-5):
        self.n_clusters = n_clusters
        self.n_epoches = n_epoches
        self.tolerance = tolerance
        self.labels = None
        self.centroids = None


    # 初始化中心点（随机选取）
    def _init_centroids(self, X: np.ndarray):
        self.centroids =  X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]


    # 计算每个点到所有中心点的距离
    def _calc_distance(self, X: np.ndarray):
        return np.sqrt(np.sum((X[:, np.newaxis] - self.centroids) ** 2, axis=2))


    # 拟合
    def fit(self, X: np.ndarray):
        """
        Args:
            X (np.ndarray): 特征数据
        """
        self._init_centroids(X)
        self.labels = np.zeros(shape=(X.shape[0]))

        for epoch in range(self.n_epoches):
            distances = self._calc_distance(X)
            self.labels = np.argmin(distances, axis=1)
            new_centroids = np.array([np.mean(X[i == self.labels], axis=0) for i in range(self.n_clusters)])

            # 判断偏移量是否小于阈值
            if np.allclose(new_centroids, self.centroids, rtol=0, atol=self.tolerance):
                break

            self.centroids = new_centroids


    def predict(self, X):
        distance = self._calc_distance(X)
        y_pred = np.argmin(distance, axis=1)

        return y_pred