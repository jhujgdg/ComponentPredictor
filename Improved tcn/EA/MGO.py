import numpy as np
from tqdm import tqdm
import torch
from tcn.criterion import BinaryCrossEntropyLoss, calculate_accuracy
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
    my_model = TemporalConvNet(num_embeddings, embedding_dim, num_channels, threshold=x[3])
    optimizer = torch.optim.Adam(my_model.parameters(), lr=x[0], weight_decay=x[2])
    bce_loss = BinaryCrossEntropyLoss()
    history = []

    for i in range(500):
        my_model.train()
        optimizer.zero_grad()  # 清零梯度
        outputs = my_model(train_input)
        loss = bce_loss.compute_loss(outputs, train_labels)

        # 反向传播
        loss.backward()
        optimizer.step()
        my_model.eval()
        # 测试集评估
        test_outputs = my_model(test_input)
        test_loss = bce_loss.compute_loss(test_outputs, test_labels)
        test_acc, test_precision, test_recall, test_F1 = calculate_accuracy(test_labels, test_outputs)

        history.append((-1 * test_F1))

    return min(history)


def initialization(SearchAgents_no, dim, ub, lb):
    return np.random.rand(SearchAgents_no, dim) * (ub - lb) + lb


def MGO(aroa_param, model_param):
    SearchAgents_no = aroa_param['N']
    MaxFEs = aroa_param['maxEvals']
    lb = aroa_param['lb']
    ub = aroa_param['ub']
    dim = aroa_param['dim']

    # Initialization
    FEs = 0
    best_cost = np.inf  # change this to -inf for maximization problems
    best_M = np.zeros(dim)
    M = initialization(SearchAgents_no, dim, ub, lb)  # Initialize the set of random solutions
    costs = np.zeros(SearchAgents_no)

    for i in range(SearchAgents_no):
        costs[i] = fobj(model_param, M[i, :])
        FEs += 1
        if costs[i] < best_cost:
            best_M = M[i, :]
            best_cost = costs[i]

    Convergence_curve = []
    it = 0
    rec = 1
    w = 2
    rec_num = 10
    divide_num = dim // 4
    d1 = 0.2

    newM = np.zeros((SearchAgents_no, dim))
    newM_cost = np.zeros(SearchAgents_no)
    rM = np.zeros((SearchAgents_no, dim, rec_num))  # record history positions
    rM_cos = np.zeros((1, SearchAgents_no, rec_num))

    # Main Loop with progress bar
    with tqdm(total=MaxFEs, desc="MGO Progress") as pbar:
        while FEs < MaxFEs:
            calPositions = M.copy()
            div_num = np.random.permutation(dim)

            # Divide the population and select the regions with more individuals based on the best
            for j in range(max(divide_num, 1)):
                th = best_M[div_num[j]]
                index = calPositions[:, div_num[j]] > th
                if np.sum(index) < calPositions.shape[0] / 2:  # choose the side of the majority
                    index = ~index
                calPositions = calPositions[index, :]

            D = best_M - calPositions  # Compute the distance from individuals to the best
            D_wind = np.sum(D, axis=0) / calPositions.shape[0]  # Calculate the mean of all distances

            beta = calPositions.shape[0] / SearchAgents_no
            gama = 1 / np.sqrt(1 - beta ** 2)
            step = w * (np.random.rand(*D_wind.shape) - 0.5) * (1 - FEs / MaxFEs)
            step2 = 0.1 * w * (np.random.rand(*D_wind.shape) - 0.5) * (1 - FEs / MaxFEs) * (
                    1 + 1 / 2 * (1 + np.tanh(beta / gama)) * (1 - FEs / MaxFEs))
            step3 = 0.1 * (np.random.rand() - 0.5) * (1 - FEs / MaxFEs)
            act = actCal(1 / (1 + (0.5 - 10 * (np.random.rand(*D_wind.shape)))))

            if rec == 1:  # record the first generation of positions
                rM[:, :, rec - 1] = M
                rM_cos[0, :, rec - 1] = costs
                rec += 1

            for i in range(SearchAgents_no):
                newM[i, :] = M[i, :]
                # Spore dispersal search
                # Update M using Eq.(6)
                if np.random.rand() > d1:
                    newM[i, :] += step * D_wind
                else:
                    newM[i, :] += step2 * D_wind

                if np.random.rand() < 0.8:
                    # Dual propagation search
                    # Update M using Eq.(11)
                    if np.random.rand() > 0.5:
                        newM[i, div_num[0]] = best_M[div_num[0]] + step3 * D_wind[div_num[0]]
                    else:
                        newM[i, :] = (1 - act) * newM[i, :] + act * best_M

                # Boundary absorption
                Flag4ub = newM[i, :] > ub
                Flag4lb = newM[i, :] < lb
                newM[i, :] = (newM[i, :] * (~(Flag4ub | Flag4lb))) + ub * Flag4ub + lb * Flag4lb
                newM_cost[i] = fobj(model_param, newM[i, :])
                FEs += 1
                # Cryptobiosis mechanism
                rM[i, :, rec - 1] = newM[i, :]
                rM_cos[0, i, rec - 1] = newM_cost[i]

                if newM_cost[i] < best_cost:
                    best_M = newM[i, :]
                    best_cost = newM_cost[i]

            rec += 1
            # Cryptobiosis mechanism
            if rec > rec_num or FEs >= MaxFEs:
                lcost = np.min(rM_cos, axis=2)
                Iindex = np.argmin(rM_cos, axis=2)
                for i in range(SearchAgents_no):
                    M[i, :] = rM[i, :, Iindex[0, i]]
                costs = lcost[0, :]
                rec = 1

            Convergence_curve.append(best_cost)
            it += 1

            # Update progress bar
            pbar.update(FEs)

    return best_cost, best_M, Convergence_curve


def actCal(X):
    act = np.where(X >= 0.5, 1, 0)
    return act
