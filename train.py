import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score, log_loss
from sklearn.ensemble import RandomForestClassifier
from imblearn.metrics import geometric_mean_score
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import umap

warnings.filterwarnings('ignore')


def load_keel_dat(file_path):
    """读取 KEEL 格式的 .dat 文件（自动跳过元信息行），并对字符型特征进行编码"""
    with open(file_path, 'r') as f:
        lines = f.readlines()
        # 找到数据部分的开始位置
        data_start = next(i for i, line in enumerate(lines) if line.strip().lower() == '@data')

    # 读取文件时不指定 dtype，让 pandas 自动推断每列的类型
    df = pd.read_csv(file_path, skiprows=data_start + 1, header=None)

    # 对字符型特征进行 Label Encoding
    label_encoder = LabelEncoder()

    # 假设所有特征列除了最后一列标签列是字符型的
    for col in df.columns[:-1]:  # 排除最后一列标签
        if df[col].dtype == 'object':  # 如果是字符型数据
            df[col] = label_encoder.fit_transform(df[col])

    # 将最后一列标签转换为数字：positive -> 1, negative -> 0
    df.iloc[:, -1] = df.iloc[:, -1].str.strip().replace({'positive': 1, 'negative': 0}).astype(np.int32)

    return df.to_numpy()


def sliding_window(input, w, s):
    """
    :param input: 输入特征数组，shape为 (n_samples, n_features)
    :param w: 窗口大小
    :param s: 步长
    :return: 特征子集数组列表，每个子集 shape 为 (n_samples, w)
    """
    n_samples, n_features = input.shape
    feature_subsets = []

    start = 0
    while start < n_features:
        # 构建窗口索引（不足则从头部补齐）
        indices = [(start + i) % n_features for i in range(w)]
        subset = input[:, indices]  # 提取实际特征子集
        feature_subsets.append(subset)
        start += s

    return feature_subsets


iter_acc_list = []
iter_prec_list = []
iter_recall_list = []
iter_f1_list = []
iter_gmean_list = []
iter_auc_list = []

k_clusters = 8
w = 6
s = 4
ratio = 5.0

for iter in range(1, 21):
    # 结果保存
    acc_list, prec_list, recall_list, f1_list, gmean_list, auc_list = [], [], [], [], [], []

    # 遍历5折数据
    for fold in range(1, 6):
        train_file = f"../five_fold_dataset/yeast6/yeast6-5-{fold}tra.dat"
        test_file = f"../five_fold_dataset/yeast6/yeast6-5-{fold}tst.dat"

        train_data = load_keel_dat(train_file)
        test_data = load_keel_dat(test_file)

        train_data = np.array(train_data, dtype=float)
        test_data = np.array(test_data, dtype=float)

        X_train_full, y_train_full = train_data[:, :-1], train_data[:, -1]

        # 比如我们划出 25% 的训练集作为验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full,
            test_size=0.25,
            stratify=y_train_full,  # 保持类分布一致
            random_state=42
        )

        X_test, y_test = test_data[:, :-1], test_data[:, -1]

        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        """ """  #################################################################
        reducer = umap.UMAP(n_components=2, random_state=42)  # 创建一个 umap 对象

        # # umap 可视化训练集、测试集多数类和少数类的分布(每一折)
        # embedding_train = reducer.fit_transform(X_train)
        # embedding_test = reducer.transform(X_test)
        #
        # # 创建左右两个子图
        # fig, axes = plt.subplots(1, 2, figsize=(14, 6))  # 1 行 2 列
        #
        # # Determine the limits for both subplots
        # x_min = min(embedding_train[:, 0].min(), embedding_test[:, 0].min())
        # x_max = max(embedding_train[:, 0].max(), embedding_test[:, 0].max())
        # y_min = min(embedding_train[:, 1].min(), embedding_test[:, 1].min())
        # y_max = max(embedding_train[:, 1].max(), embedding_test[:, 1].max())
        #
        # # Add some padding to the axis limits to ensure points near the edges are not clipped
        # padding = 0.1
        # x_range = x_max - x_min
        # y_range = y_max - y_min
        #
        # # Set limits for both subplots
        # axes[0].set_xlim(x_min - padding * x_range, x_max + padding * x_range)
        # axes[0].set_ylim(y_min - padding * y_range, y_max + padding * y_range)
        # axes[1].set_xlim(x_min - padding * x_range, x_max + padding * x_range)
        # axes[1].set_ylim(y_min - padding * y_range, y_max + padding * y_range)
        #
        # # 子图1：训练集
        # axes[0].scatter(embedding_train[y_train == 0, 0], embedding_train[y_train == 0, 1],
        #                 color='#3f72af', label='Negative class', s=15)
        # axes[0].scatter(embedding_train[y_train == 1, 0], embedding_train[y_train == 1, 1],
        #                 color='#f6416c', label='Positive class', s=15)
        # axes[0].set_title('Train Distribution')
        # axes[0].set_xlabel('Dimension-1')
        # axes[0].set_ylabel('Dimension-2')
        # axes[0].legend()
        #
        # # 子图2：测试集
        # axes[1].scatter(embedding_test[y_test == 0, 0], embedding_test[y_test == 0, 1],
        #                 color='#3f72af', label='Negative class', s=15)
        # axes[1].scatter(embedding_test[y_test == 1, 0], embedding_test[y_test == 1, 1],
        #                 color='#f6416c', label='Positive class', s=15)
        # axes[1].set_title('Test Distribution')
        # axes[1].set_xlabel('Dimension-1')
        # axes[1].set_ylabel('Dimension-2')
        # axes[1].legend()
        #
        # plt.tight_layout()
        # plt.show()
        """ """  #################################################################

        # 提取多数类和少数类
        X_train_labeled = np.hstack((X_train, y_train.reshape(-1, 1)))
        negative_class = X_train_labeled[X_train_labeled[:, -1] == 0]  # 带有标签的多数类
        positive_class = X_train_labeled[X_train_labeled[:, -1] == 1]  # 带有标签的少数类

        # 滑动窗口提取正负类样本的特征，生成多个特征子集
        negative_fs = sliding_window(negative_class[:, :-1], w=w, s=s)
        positive_fs = sliding_window(positive_class[:, :-1], w=w, s=s)

        comb_fs = []
        for a in range(len(negative_fs)):
            #
            # N_pos = positive_fs[a].shape[0]
            #
            # # Step 1: 聚类
            # kmeans = KMeans(n_clusters=k_clusters, random_state=None)
            # clusters = kmeans.fit_predict(negative_fs[a])
            #
            # # Step 2: 每簇样本数 & 占比
            # cluster_sizes = np.array([np.sum(clusters == i) for i in range(k_clusters)])
            # cluster_ratios = cluster_sizes / cluster_sizes.sum()
            #
            # # 初步配额：每簇该抽多少负类样本
            # raw_quotas = cluster_ratios * N_pos
            # quotas = np.floor(raw_quotas).astype(int)
            #
            # # 调整配额，使总和正好等于 N_pos
            # delta = N_pos - quotas.sum()
            # if delta > 0:
            #     extras = np.argsort(-raw_quotas + quotas)  # 哪些簇分配得少
            #     for i in range(delta):
            #         quotas[extras[i]] += 1
            #
            # # Step 3: 计算正类中心
            # pos_center = np.mean(positive_fs[a], axis=0, keepdims=True)
            #
            # # Step 4: 每个簇内按“与正类相似度”排序并抽取 quota 个负类样本
            # selected_samples = []
            # for cluster_id in range(k_clusters):
            #     idx = np.where(clusters == cluster_id)[0]
            #     cluster_samples = (negative_fs[a])[idx]
            #
            #     quota = quotas[cluster_id]
            #     if quota == 0 or len(idx) == 0:
            #         continue
            #
            #     # 欧式距离 + 余弦相似度
            #     eu_dist = np.linalg.norm(cluster_samples - pos_center, axis=1)
            #     cos_sim = cosine_similarity(cluster_samples, pos_center).flatten()
            #
            #     # 标准化
            #     eu_dist_norm = (eu_dist - eu_dist.min()) / (eu_dist.max() - eu_dist.min() + 1e-8)
            #     cos_sim_norm = (cos_sim - cos_sim.min()) / (cos_sim.max() - cos_sim.min() + 1e-8)
            #
            #     # 综合评分（越小越像正类）
            #     score = 0.5 * eu_dist_norm + 0.5 * (1 - cos_sim_norm)
            #
            #     # 排序并选出前 quota 个
            #     topk_local = np.argsort(score)[:quota]
            #     selected_samples.extend(cluster_samples[topk_local])
            #
            # hard_negative_samples = np.array(selected_samples)
            N_pos = len(positive_fs[a])

            # Step 1: 聚类负类
            kmeans = KMeans(n_clusters=k_clusters, random_state=None)
            clusters = kmeans.fit_predict(negative_fs[a])

            # Step 2: 计算每个簇的样本量占比
            cluster_sizes = np.array([np.sum(clusters == i) for i in range(k_clusters)])
            cluster_ratios = cluster_sizes / cluster_sizes.sum()

            # Step 3: 初始分配 quota（每簇应采样数量）
            raw_quotas = ratio * cluster_ratios * N_pos
            quotas = np.floor(raw_quotas).astype(int)

            # 调整 quota，使总数 == N_pos
            delta = N_pos - quotas.sum()
            if delta > 0:
                extras = np.argsort(-raw_quotas + quotas)  # 哪些簇分配少了
                for i in range(delta):
                    quotas[extras[i]] += 1

            # Step 4: 每个簇中随机采样
            selected_samples = []

            for cluster_id in range(k_clusters):
                idx = np.where(clusters == cluster_id)[0]
                quota = quotas[cluster_id]

                if quota == 0 or len(idx) == 0:
                    continue

                # 随机从该簇中采样 quota 个负类样本
                selected_idx = np.random.choice(idx, size=min(quota, len(idx)), replace=False)
                selected_samples.append((negative_fs[a])[selected_idx])

            # 合并所有采样结果
            balanced_neg_samples = np.vstack(selected_samples)

            neg_labels = np.zeros((balanced_neg_samples.shape[0], 1))  # shape = (n, 1)
            neg_with_labels = np.hstack((balanced_neg_samples, neg_labels))

            pos_labels = np.ones((positive_fs[a].shape[0], 1))  # shape = (n, 1)
            pos_with_labels = np.hstack((positive_fs[a], pos_labels))

            comb = np.vstack((neg_with_labels, pos_with_labels))
            comb_fs.append(comb)

        # 训练各个特征子集
        trained_model = []
        for com in comb_fs:
            model = RandomForestClassifier()
            model.fit(com[:, :-1], com[:, -1])
            trained_model.append(model)

        val_fs = sliding_window(X_val, w=w, s=s)

        val_acc = []
        val_prec = []
        val_recall = []
        val_f1 = []
        val_gmean = []
        val_auc = []
        val_loss = []

        # 验证模型
        for i in range(len(val_fs)):
            val_data = val_fs[i]
            val_model = trained_model[i]

            y_val_pred = val_model.predict(val_data)
            y_val_prob = val_model.predict_proba(val_data)[:, 1]

            val_acc.append(accuracy_score(y_val, y_val_pred))
            val_prec.append(precision_score(y_val, y_val_pred, zero_division=0))
            val_recall.append(recall_score(y_val, y_val_pred, zero_division=0))
            val_f1.append(f1_score(y_val, y_val_pred, zero_division=0))
            val_gmean.append(geometric_mean_score(y_val, y_val_pred))
            val_auc.append(roc_auc_score(y_val, y_val_prob))
            val_loss.append(log_loss(y_val, y_val_prob))

        raw_weights = [1 / (1 + loss) for loss in val_loss]

        weights = np.array(raw_weights)
        weights /= weights.sum()

        # 测试模型
        test_fs = sliding_window(X_test, w=w, s=s)

        test_prob = []
        for b in range(len(test_fs)):
            test_model_data = test_fs[b]
            test_model = trained_model[b]

            y_test_pred = test_model.predict(test_model_data)
            y_test_prob = test_model.predict_proba(test_model_data)
            test_prob.append(y_test_prob)

        fs_weighted_prob = []
        for c in range(len(weights)):
            weighted_prob = weights[c] * test_prob[c]
            fs_weighted_prob.append(weighted_prob)

        final_prob = sum(fs_weighted_prob)
        pred_labels = (final_prob[:, 1] > final_prob[:, 0]).astype(int)

        acc_list.append(accuracy_score(y_test, pred_labels))
        prec_list.append(precision_score(y_test, pred_labels, zero_division=0))
        recall_list.append(recall_score(y_test, pred_labels, zero_division=0))
        f1_list.append(f1_score(y_test, pred_labels, zero_division=0))
        gmean_list.append(geometric_mean_score(y_test, pred_labels))
        auc_list.append(roc_auc_score(y_test, final_prob[:, 1]))
    pass

    print(f"Iteration {iter}")
    print(f"Accuracy = {np.mean(acc_list):.4f}")
    print(f"F1-score = {np.mean(f1_list):.4f}")
    print(f"ROC-AUC = {np.mean(auc_list):.4f}")
    print(" ")

    pass
