import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_excel_file(file_path):
    """加载Excel文件"""
    try:
        return pd.read_excel(file_path)
    except Exception as e:
        print(f"加载文件失败: {e}")
        return None

def calculate_euclidean_distance(vec1, vec2):
    """计算欧氏距离"""
    return np.sqrt(np.sum((vec1 - vec2) ** 2))

def get_nearest_neighbors(target, candidates, k=3):
    """获取最近的k个邻居"""
    distances = []
    for i, candidate in enumerate(candidates):
        distance = calculate_euclidean_distance(target, candidate)
        distances.append((i, distance))
    distances.sort(key=lambda x: x[1])
    return [i for i, _ in distances[:k]]