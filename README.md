# MyMLModule - 机器学习基础模块

`MyMLModule` 是一个用于学习机器学习的 Python 模块。  
我在学习机器学习的过程中基于[numpy](https://github.com/numpy/numpy)手写了一些基本的机器学习算法，主要目的是**加深对机器学习原理的理解**。

## 🚀 安装
### **使用 `pip` 安装**
```bash
pip install git+https://github.com/TGMax233Michael/MyMLModule.git
```

## 🛠 使用示例

```python
from MyMLModule.models.linear_model import LinearRegression
import numpy as np

# 创建数据
X = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])

# 训练模型
model = LinearRegression()
model.fit(X, y)

# 预测
print(model.predict(np.array([6])))  # 输出预测值
```

## 📌 目前实现的功能
- ✅ 线性回归 Linear Regression
- ✅ 岭回归 Ridge Regression
- ✅ 逻辑回归 Logistic Regression
- ✅ 决策树 Decision Tree
- ✅ K-Means 聚类 K-Means Clustering
- ⚠️ 支持向量机 SVM (暂不支持核函数 Kernel not supported)
