import numpy as np
from scipy.special import gamma
from tqdm import tqdm
import torch
from pygcn import accuracy
from ..model import TemporalConvNet
from ..criterion import BinaryCrossEntropyLoss, calculate_accuracy


def fobj(model_param, x):
    num_embeddings = model_param['num_embeddings']
    y_train = model_param['y_train']
    train_input = model_param['train_input']
    test_input = model_param['test_input']
    train_labels = model_param['train_labels']
    test_labels = model_param['test_labels']

    # 模型参数
    embedding_dim = int(x[1])
    num_channels = [16, 32, 64]
    my_model = TemporalConvNet(num_embeddings, embedding_dim, num_channels,threshold=x[3])
    optimizer = torch.optim.Adam(my_model.parameters(), lr=x[0], weight_decay=x[2])
    # criterion = get_criterion("BCELoss", num_embeddings, weight)
    bce_loss = BinaryCrossEntropyLoss()
    history = []

    for i in range(500):
        my_model.train()
        optimizer.zero_grad()  # 清零梯度
        outputs= my_model(train_input)
        loss = bce_loss.compute_loss(outputs, train_labels)

        # 反向传播
        loss.backward()
        optimizer.step()
        my_model.eval()
        # 测试集评估
        test_outputs = my_model(test_input)  # 修改这一行以获取模型输出和注意力权重
        test_loss = bce_loss.compute_loss(test_outputs, test_labels)
        test_acc, test_precision, test_recall, test_F1 = calculate_accuracy(test_labels, test_outputs)

        history.append((-1 * test_F1))

    return min(history)


def levy(n, m, beta):
    num = gamma(1 + beta) * np.sin(np.pi * beta / 2)
    den = gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2)
    sigma_u = (num / den) ** (1 / beta)

    u = np.random.normal(0, sigma_u, (n, m))
    v = np.random.normal(0, 1, (n, m))

    return u / (np.abs(v) ** (1 / beta))


def initialization(SearchAgents_no, dim, ub, lb):
    return np.random.uniform(low=lb, high=ub, size=(SearchAgents_no, dim))


def GJO(aroa_param, model_param):
    SearchAgents_no = aroa_param['N']
    Max_iter = aroa_param['maxEvals']
    lb = aroa_param['lb']
    ub = aroa_param['ub']
    dim = aroa_param['dim']
    Male_Jackal_pos = np.zeros(dim)
    Male_Jackal_score = float('inf')
    Female_Jackal_pos = np.zeros(dim)
    Female_Jackal_score = float('inf')

    Positions = initialization(SearchAgents_no, dim, ub, lb)
    Convergence_curve = np.zeros(Max_iter)

    # 使用 tqdm 创建进度条
    for l in tqdm(range(Max_iter), desc="Iterations"):
        for i in range(Positions.shape[0]):
            Positions[i, :] = np.clip(Positions[i, :], lb, ub)
            fitness = fobj(model_param, Positions[i, :])

            if fitness < Male_Jackal_score:
                Male_Jackal_score = fitness
                Male_Jackal_pos = Positions[i, :]

            if Male_Jackal_score < fitness < Female_Jackal_score:
                Female_Jackal_score = fitness
                Female_Jackal_pos = Positions[i, :]

        E1 = 1.5 * (1 - (l / Max_iter))
        RL = 0.05 * levy(SearchAgents_no, dim, 1.5)

        for i in range(Positions.shape[0]):
            for j in range(Positions.shape[1]):
                r1 = np.random.rand()
                E0 = 2 * r1 - 1
                E = E1 * E0

                if abs(E) < 1:
                    D_male_jackal = abs((RL[i, j] * Male_Jackal_pos[j] - Positions[i, j]))
                    Male_Positions = Male_Jackal_pos[j] - E * D_male_jackal
                    D_female_jackal = abs((RL[i, j] * Female_Jackal_pos[j] - Positions[i, j]))
                    Female_Positions = Female_Jackal_pos[j] - E * D_female_jackal
                else:
                    D_male_jackal = abs((Male_Jackal_pos[j] - RL[i, j] * Positions[i, j]))
                    Male_Positions = Male_Jackal_pos[j] - E * D_male_jackal
                    D_female_jackal = abs((Female_Jackal_pos[j] - RL[i, j] * Positions[i, j]))
                    Female_Positions = Female_Jackal_pos[j] - E * D_female_jackal

                Positions[i, j] = (Male_Positions + Female_Positions) / 2

        # 记录当前的最佳适应度值
        Convergence_curve[l] = Male_Jackal_score

    return Male_Jackal_score, Male_Jackal_pos, Convergence_curve

# 定义目标函数
# def objective_function(position):
#     # 这里使用一个简单的目标函数，例如求和平方
#     return np.sum(position**2)
#
# # 设置 GJO 参数
# SearchAgents_no = 30  # 种群个体数量
# Max_iter = 100        # 最大迭代次数
# lb = np.array([-10, -10, -10, -10])  # 下界
# ub = np.array([10, 10, 10, 10])       # 上界
# dim = 4  # 维度数量
#
# # 调用 GJO 函数
# Male_Jackal_score, Male_Jackal_pos, Convergence_curve = GJO(SearchAgents_no, Max_iter, lb, ub, dim, objective_function)
#
# # 输出结果
# print("最优解的适应度值:", Male_Jackal_score)
# print("最优解的位置:", Male_Jackal_pos)
# print("收敛曲线:", Convergence_curve)
