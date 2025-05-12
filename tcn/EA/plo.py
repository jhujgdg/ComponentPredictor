import numpy as np
from scipy.special import gamma
from tqdm import tqdm
import torch
from pygcn import accuracy
from tcn.model import TemporalConvNet
from tcn.criterion import BinaryCrossEntropyLoss, calculate_accuracy


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


def initialization(N, dim, ub, lb):
    return np.random.uniform(low=lb, high=ub, size=(N, dim))


def Levy(d):
    beta = 1.5
    sigma = (gamma(1 + beta) * np.sin(np.pi * beta / 2) /
             (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = np.random.normal(0, sigma, d)
    v = np.random.normal(0, 1, d)
    return u / np.abs(v) ** (1 / beta)


def PLO(param, model_param):
    N, dim, lb, ub, MaxFEs = param['N'], param['dim'], param['lb'], param['ub'], param['maxEvals']

    FEs = 0
    fitness = np.inf * np.ones(N)
    X = initialization(N, dim, ub, lb)

    # 计算初始适应度
    for i in range(N):
        fitness[i] = fobj(model_param, X[i, :])
        FEs += 1

    # 排序适应度
    fitness, SortOrder = np.sort(fitness), np.argsort(fitness)
    X = X[SortOrder, :]
    Bestpos, Bestscore = X[0, :], fitness[0]

    Convergence_curve = [Bestscore]

    while FEs <= MaxFEs:
        X_mean = np.mean(X, axis=0)
        w1 = np.tanh((FEs / MaxFEs) ** 4)
        w2 = np.exp(-(2 * FEs / MaxFEs) ** 3)

        for i in range(N):
            a = np.random.rand() / 2 + 1
            LS = np.ones(dim) * np.exp((1 - a) / 100 * FEs)
            GS = Levy(dim) * (X_mean - X[i, :] + (lb + np.random.rand(dim) * (ub - lb)) / 2)
            X_new = X[i, :] + (w1 * LS + w2 * GS) * np.random.rand(dim)

            # 边界检查
            X_new = np.clip(X_new, lb, ub)
            fitness_new = fobj(model_param, X_new)
            FEs += 1

            if fitness_new < fitness[i]:
                X[i, :], fitness[i] = X_new, fitness_new

        # 更新最佳解
        fitness, SortOrder = np.sort(fitness), np.argsort(fitness)
        X = X[SortOrder, :]
        if fitness[0] < Bestscore:
            Bestpos, Bestscore = X[0, :], fitness[0]

        Convergence_curve.append(Bestscore)

    return Bestpos, Bestscore, Convergence_curve
