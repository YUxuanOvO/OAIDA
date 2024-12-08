import numpy as np



def remove_linearly_related_features(data):
    # 计算特征之间的相关性矩阵
    corr_matrix = data.corr().abs()

    # 创建一个布尔矩阵，标记是否特征对之间具有高于阈值的相关性
    threshold = 0.9  # 设置相关性阈值，根据需要调整
    upper_tri = np.triu(np.ones(corr_matrix.shape), k=1)
    correlated_features = (corr_matrix > threshold) & upper_tri

    # 找到要删除的特征列，但保留名为"Label"的列
    columns_to_drop = set()
    for col in correlated_features.columns:
        correlated_cols = list(correlated_features.index[correlated_features[col]])
        if correlated_cols:
            columns_to_drop.add(col)
            columns_to_drop.update(correlated_cols)

    # 从数据中删除冗余特征列，但保留名为"Label"的列
    columns_to_drop.discard("Label")
    data_cleaned = data.drop(columns=columns_to_drop)

    return data_cleaned