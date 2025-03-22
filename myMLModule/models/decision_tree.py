import numpy as np

"""
    计算用函数区
"""
# 计算基尼指数
def _gini(y: np.ndarray):
    labels, counts = np.unique(y, return_counts=True)
    ratio = counts / len(y)
    return 1 - np.sum(ratio**2)

# 计算均方误差
def _mse(y: np.ndarray):
    if len(y) == 0:
        return 0
    return 1/len(y) * np.sum((y - np.mean(y))**2)


# 树节点
class DecisionTreeNode:
    """
        决策树节点

        属性:
            feature: 选取的最佳特征
            threshold: 对应的最佳阈值
            left: 左节点
            right: 右节点
            label: 标签（仅当前节点为叶节点的时候才会有标签）
    """
    def __init__(self, feature=None, threshold=None, left=None, right=None, label=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.label = label



"""
    模型区
"""
class TreeClassifier:
    """
        分类决策树
        属性:
            max_depth: 最深层数
            min_batch: 分割时节点最少需要的样本量（小于该阈值则直接将该节点作为叶节点）
            gain_threshold: 信息增益的阈值（小于该阈值则直接将该节点作为叶节点）
    """
    def __init__(self, max_depth=100, min_batch=2, gain_threshold=0.01):
        self.tree: DecisionTreeNode|None = None
        self.max_depth = max_depth
        self.min_batch = min_batch
        self.gain_threshold = gain_threshold


    # 获取该子集最佳划分方式
    def _best_split(self, X: np.ndarray, y: np.ndarray):
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # 初始化数据
        # 使用基尼系数作为信息熵
        n_samples, n_features = X.shape
        best_threshold = None
        best_feature = None
        base_gini = _gini(y)
        best_gain = -np.inf

        for feature in range(n_features):
            sorted_uniques = np.sort(np.unique(X[:, feature]))
            if len(sorted_uniques) == 1:
                thresholds = sorted_uniques
            else:
                thresholds = (sorted_uniques[:-1] + sorted_uniques[1:]) / 2

            for threshold in thresholds:
                left_mask = X[:, feature] < threshold
                right_mask = ~left_mask

                y_left = y[left_mask]
                y_right = y[right_mask]

                weighted_gini = len(y_left)/n_samples * _gini(y_left) + len(y_right)/n_samples * _gini(y_right)
                gain = base_gini - weighted_gini


                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_gain, best_feature, best_threshold


    def _build_tree(self, X: np.ndarray, y: np.ndarray, layer):
        gain, feature, threshold = self._best_split(X, y)
        if len(np.unique(y)) == 1:
            return DecisionTreeNode(label=y[0])

        # 达到预设深度 || 预设最小集合大小 || 信息增益达到阈值
        if layer == self.max_depth or len(y) <= self.min_batch or gain <= self.gain_threshold:
            # 将出现最多的类别数作为标签
            return DecisionTreeNode(label=np.argmax(np.bincount(y)))

        left_mask = X[:, feature] < threshold
        right_mask = ~left_mask
        left = self._build_tree(X[left_mask], y[left_mask], layer+1)
        right = self._build_tree(X[right_mask], y[right_mask], layer+1)

        return DecisionTreeNode(feature, threshold, left, right)


    def _travel_tree(self, X: np.ndarray, node: DecisionTreeNode, layer):
        if node is None:
            return None
        if node.label is not None:
            return np.full(X.shape[0], node.label)

        predictions = np.empty(X.shape[0], dtype=np.float64)
        left_mask = X[:, node.feature] < node.threshold
        right_mask = ~left_mask

        predictions[left_mask] = self._travel_tree(X[left_mask], node.left, layer+1)
        predictions[right_mask] = self._travel_tree(X[right_mask], node.right, layer+1)

        return predictions


    # 拟合
    def fit(self, X: np.ndarray, y: np.ndarray):
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.tree = self._build_tree(X, y, 1)


    # 预测
    def predict(self, X: np.ndarray):
        return self._travel_tree(X, self.tree, 0)


class TreeRegressor:
    def __init__(self, max_depth=100, min_batch=10, gain_threshold=0):
        self.tree: DecisionTreeNode|None = None
        self.max_depth = max_depth
        self.min_batch = min_batch
        self.gain_threshold = gain_threshold


    # 获取该子集最佳划分方式
    def _best_split(self, X: np.ndarray, y: np.ndarray):
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # 数据初始化
        # 使用MSE作为信息熵
        n_samples, n_featurs = X.shape
        best_feature = None
        best_threshold = None
        best_gain = -np.inf
        base_MSE = _mse(y)

        for feature in range(n_featurs):
            sorted_unique = np.sort(np.unique(X[:, feature]))
            if len(sorted_unique) == 1:
                thresholds = sorted_unique
            else:
                thresholds = (sorted_unique[:-1] + sorted_unique[1:]) / 2

            for threshold in thresholds:
                left_mask = X[:, feature] < threshold
                right_mask = ~left_mask

                weighted_MSE = len(y[left_mask])/len(y) * _mse(y[left_mask]) + len(y[right_mask])/len(y) * _mse(y[right_mask])
                gain = base_MSE - weighted_MSE

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_gain, best_feature, best_threshold


    def _build_tree(self, X: np.ndarray, y: np.ndarray, layer):
        gain, feature, threshold = self._best_split(X, y)

        if len(y) == 0:
            return None
        if len(np.unique(y)) == 1:
            return DecisionTreeNode(label=y[0])

        # 达到预设深度 || 预设最小集合大小 || 信息增益达到阈值
        if layer >= self.max_depth or len(y) <= self.min_batch or gain <= self.gain_threshold:
            return DecisionTreeNode(label=np.mean(y))

        left_mask = X[:, feature] < threshold
        right_mask = ~left_mask

        left = self._build_tree(X[left_mask], y[left_mask], layer+1)
        right = self._build_tree(X[right_mask], y[right_mask], layer+1)

        return DecisionTreeNode(feature, threshold, left, right)


    def _travel_tree(self, X: np.ndarray, node: DecisionTreeNode):
        if node is None or len(X) == 0:
            return None
        if node.label is not None:
            return np.full(X.shape[0], node.label)

        predictions = np.empty(X.shape[0], dtype=np.float64)
        left_mask = X[:, node.feature] < node.threshold
        right_mask = ~left_mask

        predictions[left_mask] = self._travel_tree(X[left_mask], node.left)
        predictions[right_mask] = self._travel_tree(X[right_mask], node.right)

        return predictions


    # 拟合
    def fit(self, X: np.ndarray, y: np.ndarray):
        self.tree = self._build_tree(X, y, 1)


    # 预测
    def predict(self, X: np.ndarray):
        return self._travel_tree(X, self.tree)