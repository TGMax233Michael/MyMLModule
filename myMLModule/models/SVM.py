import numpy as np

class SVM_binary:
    def __init__(self, epoches=100, learning_rate=0.01, C=1, bias: bool=True):
        self.epoches = epoches
        self.learning_rate = learning_rate
        self.C = C
        self._coef = None
        self.bias = bias


    def _init_coef(self, n_features):
        self._coef = np.zeros(n_features)


    def fit(self, X: np.ndarray, y: np.ndarray):

        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X 与 y 的样本数量不匹配 X: {X.shape}, y: {y.shape}")

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # 将 0 标签改为 -1，适用于SVM
        y[y == 0] = -1


        if self.bias:
            X = np.column_stack((np.ones(X.shape[0]), X))

        n_samples, n_features = X.shape
        self._init_coef(n_features)

        for epoch in range(self.epoches):
            y_pred = X @ self._coef
            mask = y * (y_pred) < 1
            loss = np.sum(1 - y[mask] * (X[mask] @ self._coef))
            gradient = self._coef - self.C * np.sum(y[mask].reshape(-1, 1) * X[mask], axis=0)
            self._coef -= gradient * self.learning_rate


    def predict(self, X):
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if self.bias:
            X = np.column_stack((np.ones(X.shape[0]), X))

        y_pred = X @ self._coef
        return y_pred



# 多分类使用 ova
class SVM:
    def __init__(self, epoches=100, learning_rate=0.01, C=1, bias: bool=True):
        self.epoches = epoches
        self.learning_rate = learning_rate
        self.C = C
        self.bias = bias
        self._coef = None
        self.classes = None
        self.classifiers: dict[any, SVM_binary] = {}


    def fit(self, X: np.ndarray, y: np.ndarray):
        self.classes = np.unique(y)

        for c in self.classes:
            y_binary = (y==c).astype(int)
            classifier = SVM_binary(epoches=self.epoches,
                                    learning_rate=self.learning_rate,
                                    C=self.C,
                                    bias=self.bias)

            classifier.fit(X, y_binary)
            self.classifiers[c] = classifier


    def predict(self, X):
        n_samples = X.shape[0]

        probabilitites = np.zeros(shape=(n_samples, len(self.classes)))

        for idx, c in enumerate(self.classes):
            probabilitites[:, idx] = self.classifiers[c].predict(X)

        y_pred = np.argmax(probabilitites, axis=1)

        return y_pred