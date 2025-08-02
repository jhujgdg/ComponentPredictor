import numpy as np
from scipy.special import gamma
from tqdm import tqdm
import torch
from pygcn import accuracy
from ..criterion import BinaryCrossEntropyLoss, calculate_accuracy
from ..model import TemporalConvNet


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



def GOA(param, model_param):
    SearchAgents_no = param['N']
    Max_iter = param['maxEvals']
    lb = param['lb']
    ub = param['ub']
    dim = param['dim']
    Top_gazelle_pos = np.zeros(dim)
    Top_gazelle_fit = float('inf')

    Convergence_curve = np.zeros(Max_iter)
    stepsize = np.zeros((SearchAgents_no, dim))
    fitness = np.full(SearchAgents_no, float('inf'))

    gazelle = initialization(SearchAgents_no, dim, ub, lb)

    Xmin = np.tile(lb, (SearchAgents_no, 1))
    Xmax = np.tile(ub, (SearchAgents_no, 1))

    Iter = 0
    PSRs = 0.34
    S = 0.88
    s = np.random.rand()
    with tqdm(total=Max_iter, desc="Iterations Progress") as pbar:
        while Iter < Max_iter:
            # Evaluating top gazelle
            for i in range(gazelle.shape[0]):
                Flag4ub = gazelle[i, :] > ub
                Flag4lb = gazelle[i, :] < lb
                gazelle[i, :] = (gazelle[i, :] * (~(Flag4ub | Flag4lb))) + ub * Flag4ub + lb * Flag4lb

                fitness[i] = fobj(model_param,gazelle[i, :])

                if fitness[i] < Top_gazelle_fit:
                    Top_gazelle_fit = fitness[i]
                    Top_gazelle_pos = gazelle[i, :]

            # Keeping track of fitness values
            if Iter == 0:
                fit_old = fitness.copy()
                Prey_old = gazelle.copy()

            Inx = fit_old < fitness
            Indx = np.tile(Inx, (dim, 1)).T
            gazelle = Indx * Prey_old + ~Indx * gazelle
            fitness = Inx * fit_old + ~Inx * fitness

            fit_old = fitness.copy()
            Prey_old = gazelle.copy()

            Elite = np.tile(Top_gazelle_pos, (SearchAgents_no, 1))
            CF = (1 - Iter / Max_iter) ** (2 * Iter / Max_iter)

            RL = 0.05 * levy(SearchAgents_no, dim, 1.5)  # Levy random number vector
            RB = np.random.randn(SearchAgents_no, dim)  # Brownian random number vector

            for i in range(gazelle.shape[0]):
                for j in range(gazelle.shape[1]):
                    R = np.random.rand()
                    r = np.random.rand()
                    mu = -1 if Iter % 2 == 0 else 1

                    # Exploitation
                    if r > 0.5:
                        stepsize[i, j] = RB[i, j] * (Elite[i, j] - RB[i, j] * gazelle[i, j])
                        gazelle[i, j] += s * R * stepsize[i, j]
                    else:
                        # Exploration
                        if i > gazelle.shape[0] / 2:
                            stepsize[i, j] = RB[i, j] * (RL[i, j] * Elite[i, j] - gazelle[i, j])
                            gazelle[i, j] = Elite[i, j] + S * mu * CF * stepsize[i, j]
                        else:
                            stepsize[i, j] = RL[i, j] * (Elite[i, j] - RL[i, j] * gazelle[i, j])
                            gazelle[i, j] += S * mu * R * stepsize[i, j]

            # Updating top gazelle
            for i in range(gazelle.shape[0]):
                Flag4ub = gazelle[i, :] > ub
                Flag4lb = gazelle[i, :] < lb
                gazelle[i, :] = (gazelle[i, :] * (~(Flag4ub | Flag4lb))) + ub * Flag4ub + lb * Flag4lb

                fitness[i] = fobj(model_param,gazelle[i, :])

                if fitness[i] < Top_gazelle_fit:
                    Top_gazelle_fit = fitness[i]
                    Top_gazelle_pos = gazelle[i, :]

            # Updating history of fitness values
            if Iter == 0:
                fit_old = fitness.copy()
                Prey_old = gazelle.copy()

            Inx = fit_old < fitness
            Indx = np.tile(Inx, (dim, 1)).T
            gazelle = Indx * Prey_old + ~Indx * gazelle
            fitness = Inx * fit_old + ~Inx * fitness

            # 更新历史适应度值
            fit_old = fitness.copy()
            Prey_old = gazelle.copy()

            # Applying PSRs
            if np.random.rand() < PSRs:
                U = np.random.rand(SearchAgents_no, dim) < PSRs
                gazelle += CF * ((Xmin + np.random.rand(SearchAgents_no, dim) * (Xmax - Xmin)) * U)
            else:
                r = np.random.rand()
                Rs = gazelle.shape[0]
                stepsize = (PSRs * (1 - r) + r) * (
                        gazelle[np.random.permutation(Rs), :] - gazelle[np.random.permutation(Rs), :])
                gazelle += stepsize

            # 记录当前迭代的最佳适应度
            Convergence_curve[Iter] = Top_gazelle_fit
            Iter += 1
            pbar.update(1)

    return Top_gazelle_fit, Top_gazelle_pos, Convergence_curve


def initialization(SearchAgents_no, dim, ub, lb):
    return np.random.rand(SearchAgents_no, dim) * (ub - lb) + lb


def levy(n, m, beta):
    # 计算分子和分母
    num = gamma(1 + beta) * np.sin(np.pi * beta / 2)  # Numerator
    den = gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2)  # Denominator

    sigma_u = (num / den) ** (1 / beta)  # Standard deviation

    # 生成随机数
    u = np.random.normal(0, sigma_u, (n, m))
    v = np.random.normal(0, 1, (n, m))

    z = u / (np.abs(v) ** (1 / beta))

    return z
