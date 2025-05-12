from tqdm import tqdm
import numpy as np
import torch
from tcn.model import TemporalConvNet
from tcn.criterion import BinaryCrossEntropyLoss, calculate_accuracy


def F_obj(model_param, x):
    num_embeddings = model_param['num_embeddings']
    y_train = model_param['y_train']
    train_input = model_param['train_input']
    test_input = model_param['test_input']
    train_labels = model_param['train_labels']
    test_labels = model_param['test_labels']
    print(x)
    # 模型参数
    embedding_dim = int(x[0][1])
    num_channels = [16, 32, 64]
    my_model = TemporalConvNet(num_embeddings, embedding_dim, num_channels, threshold=x[0][3])
    optimizer = torch.optim.Adam(my_model.parameters(), lr=x[0][0], weight_decay=x[0][2])
    bce_loss = BinaryCrossEntropyLoss()
    history = []

    for i in range(10):
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


def initialization(nPop, lb, ub, nVar):
    return np.array([np.random.uniform(lb, ub) for _ in range(nPop)])


def roulette_wheel_selection(P):
    r = np.random.rand()
    C = np.cumsum(P)
    i = np.where(r <= C)[0][0]
    return i


def DMOA(aroa_param, model_param):
    nPop = aroa_param['N']
    MaxIt = aroa_param['maxEvals']
    VarMin = aroa_param['lb']
    VarMax = aroa_param['ub']
    nVar = aroa_param['dim']
    VarSize = (1, nVar)

    empty_mongoose = {
        "Position": None,
        "Cost": None
    }

    pop = [empty_mongoose.copy() for _ in range(nPop - 3)]

    BestSol = {
        "Cost": np.inf,
        "Position": None
    }
    tau = np.inf
    Iter = 1
    sm = np.full((nPop - 3, 1), np.inf)

    for i in range(nPop - 3):
        pop[i]["Position"] = np.random.uniform(VarMin, VarMax, VarSize)
        pop[i]["Cost"] = F_obj(model_param, pop[i]["Position"])
        if pop[i]["Cost"] <= BestSol["Cost"]:
            BestSol = pop[i].copy()

    C = np.zeros((nPop - 3, 1))
    CF = (1 - Iter / MaxIt) ** (2 * Iter / MaxIt)

    BestCost = np.zeros((MaxIt, 1))

    # 添加进度条
    for it in tqdm(range(MaxIt), desc="Iterations"):
        F = np.zeros((nPop - 3, 1))
        costs = np.array([p["Cost"] for p in pop])
        MeanCost = np.mean(costs)
        for i in range(nPop - 3):
            F[i] = np.exp(-pop[i]["Cost"] / MeanCost)
        P = F / np.sum(F)

        for m in range(nPop - 3):
            i = roulette_wheel_selection(P)
            K = list(range(nPop - 3))
            del K[i]
            k = np.random.choice(K)
            phi = (2 / 2) * np.random.uniform(-1, 1, VarSize)

            newpop = {
                "Position": pop[i]["Position"] + phi * (pop[i]["Position"] - pop[k]["Position"]),
                "Cost": None
            }

            newpop["Cost"] = F_obj(model_param, newpop["Position"])

            if newpop["Cost"] <= pop[i]["Cost"]:
                pop[i] = newpop.copy()
            else:
                C[i] += 1

        for i in range(nPop - 3):
            K = list(range(nPop - 3))
            del K[i]
            k = np.random.choice(K)
            phi = (2 / 2) * np.random.uniform(-1, 1, VarSize)

            newpop = {
                "Position": pop[i]["Position"] + phi * (pop[i]["Position"] - pop[k]["Position"]),
                "Cost": None
            }

            newpop["Cost"] = F_obj(model_param, newpop["Position"])
            sm[i] = (newpop["Cost"] - pop[i]["Cost"]) / max(newpop["Cost"], pop[i]["Cost"])

            if newpop["Cost"] <= pop[i]["Cost"]:
                pop[i] = newpop.copy()
            else:
                C[i] += 1

        for i in range(3):
            if C[i] >= 0.6 * nVar * 3:
                pop[i]["Position"] = np.random.uniform(VarMin, VarMax, VarSize)
                pop[i]["Cost"] = F_obj(model_param, pop[i]["Position"])
                C[i] = 0

        for i in range(nPop - 3):
            if pop[i]["Cost"] <= BestSol["Cost"]:
                BestSol = pop[i].copy()

        newtau = np.mean(sm)
        for i in range(nPop - 3):
            M = (pop[i]["Position"] * sm[i]) / pop[i]["Position"]
            if newtau > tau:
                newpop = {
                    "Position": pop[i]["Position"] - CF * phi * np.random.rand() * (pop[i]["Position"] - M),
                    "Cost": None
                }
            else:
                newpop = {
                    "Position": pop[i]["Position"] + CF * phi * np.random.rand() * (pop[i]["Position"] - M),
                    "Cost": None
                }
            tau = newtau

        for i in range(nPop - 3):
            if pop[i]["Cost"] <= BestSol["Cost"]:
                BestSol = pop[i].copy()

        BestCost[it] = BestSol["Cost"]
        BEF = BestSol["Cost"]
        BEP = BestSol["Position"]

        print(f"Iteration {it + 1}: Best Cost = {BestCost[it]}")

    return BEF, BEP, BestCost


def roulette_wheel_selection(P):
    r = np.random.rand()
    C = np.cumsum(P)
    i = np.where(r <= C)[0][0]
    return i
