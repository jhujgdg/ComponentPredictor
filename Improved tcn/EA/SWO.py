import numpy as np
from scipy.special import gamma
from tqdm import tqdm
import torch
from pygcn import accuracy
from tcn.model import TemporalConvNet
from tcn.criterion import BinaryCrossEntropyLoss, calculate_accuracy


def feval(model_param, x):
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

def levy(d):
    beta = 3 / 2
    sigma = (gamma(1 + beta) * np.sin(np.pi * beta / 2) /
             (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = np.random.randn(d) * sigma
    v = np.random.randn(d)
    step = u / np.abs(v) ** (1 / beta)
    return 0.05 * step


def initialization(search_agents_no, dim, ub, lb):
    return np.random.rand(search_agents_no, dim) * (ub - lb) + lb


def SWO(aroa_param, model_param):
    search_agents_no = aroa_param['N']
    Tmax = aroa_param['maxEvals']
    lb = aroa_param['lb']
    ub = aroa_param['ub']
    dim = aroa_param['dim']
    Best_SW = np.zeros(dim)  # Best-so-far spider wasp (solution)
    Best_score = float('inf')  # Best-so-far score
    Convergence_curve = np.zeros(Tmax)

    # Ensure lb and ub are numpy arrays
    lb = np.array(lb)
    ub = np.array(ub)

    # Controlling parameters
    TR = 0.3  # Trade-off probability between hunting and mating behaviors
    Cr = 0.2  # Crossover probability
    N_min = 20  # Minimum population size

    # Initialization
    Positions = initialization(search_agents_no, dim, ub, lb)  # Initialize positions
    t = 0  # Function evaluation counter
    SW_Fit = np.zeros(search_agents_no)

    # Evaluation
    for i in range(search_agents_no):
        SW_Fit[i] = feval(model_param, Positions[i, :])
        if SW_Fit[i] < Best_score:  # Change this to > for maximization problem
            Best_score = SW_Fit[i]
            Best_SW = Positions[i, :]

    # Main loop with progress bar
    with tqdm(total=Tmax, desc="SWO Progress") as pbar:
        while t < Tmax:
            a = 2 - 2 * (t / Tmax)  # a decreases linearly from 2 to 0
            a2 = -1 + -1 * (t / Tmax)  # a2 decreases linearly from -1 to -2
            k = (1 - t / Tmax)  # k decreases linearly from 1 to 0
            JK = np.random.permutation(search_agents_no)  # Random permutation of indices

            if np.random.rand() < TR:  # Hunting and nesting behavior
                for i in range(search_agents_no):
                    r1, r2, r3 = np.random.rand(3)
                    p = np.random.rand()
                    C = a * (2 * r1 - 1)  # Eq. (11)
                    l = (a2 - 1) * np.random.rand() + 1  # The parameter in Eqs. (7) and (8)
                    L = levy(1)  # Levy-based number
                    vc = np.random.uniform(-k, k, dim)  # The vector in Eq. (12)
                    rn1 = np.random.randn()  # Normal distribution-based number

                    O_P = Positions[i, :]  # Store the current position

                    for j in range(dim):
                        if i < k * search_agents_no:
                            if p < (1 - t / Tmax):  # Searching stage (Exploration)
                                if r1 < r2:
                                    m1 = np.abs(rn1) * r1  # Eq. (5)
                                    Positions[i, j] += m1 * (Positions[JK[0], j] - Positions[JK[1], j])  # Eq. (4)
                                else:
                                    B = 1 / (1 + np.exp(l))  # Eq. (8)
                                    m2 = B * np.cos(l * 2 * np.pi)  # Eq. (7)
                                    Positions[i, j] = Positions[JK[i], j] + m2 * (
                                                lb[j] + np.random.rand() * (ub[j] - lb[j]))  # Eq. (6)
                            else:  # Following and escaping stage (exploration and exploitation)
                                if r1 < r2:
                                    Positions[i, j] += C * np.abs(
                                        2 * np.random.rand() * Positions[JK[2], j] - Positions[i, j])  # Eq. (10)
                                else:
                                    Positions[i, j] *= vc[j]  # Eq. (12)
                        else:
                            if r1 < r2:
                                Positions[i, j] = Best_SW[j] + np.cos(2 * l * np.pi) * (
                                            Best_SW[j] - Positions[i, j])  # Eq. (16)
                            else:
                                Positions[i, j] = Positions[JK[0], j] + r3 * np.abs(L) * (
                                            Positions[JK[0], j] - Positions[i, j]) + (1 - r3) * (
                                                      np.random.rand() > np.random.rand()) * (
                                                      Positions[JK[2], j] - Positions[JK[1], j])  # Eq. (17)

                    # Return the search agents that exceed the search space's bounds
                    for j in range(dim):
                        if Positions[i, j] > ub[j]:
                            Positions[i, j] = lb[j] + np.random.rand() * (ub[j] - lb[j])
                        elif Positions[i, j] < lb[j]:
                            Positions[i, j] = lb[j] + np.random.rand() * (ub[j] - lb[j])

                    SW_Fit1 = feval(model_param, Positions[i, :])  # The fitness value of the newly generated spider
                    # Memory Saving and Updating the best-so-far solution
                    if SW_Fit1 < SW_Fit[i]:  # Change this to > for maximization problem
                        SW_Fit[i] = SW_Fit1  # Update the local best fitness
                        # Update the best-so-far solution
                        if SW_Fit[i] < Best_score:  # Change this to > for maximization problem
                            Best_score = SW_Fit[i]  # Update best-so-far fitness
                            Best_SW = Positions[i, :]  # Update best-so-far position
                    else:
                        Positions[i, :] = O_P  # Return the last best solution obtained by the ith solution

                    t += 1
                    if t >= Tmax:
                        break

                    Convergence_curve[t] = Best_score

            else:  # Mating behavior
                for i in range(search_agents_no):
                    l = (a2 - 1) * np.random.rand() + 1  # The parameter in Eqs. (7) and (8)
                    SW_m = np.zeros(dim)  # Including the spider wasp male
                    O_P = Positions[i, :]  # Store the current position

                    # The Step sizes used to generate the male spider with high quality
                    if SW_Fit[JK[0]] < SW_Fit[i]:  # Eq. (23)
                        v1 = Positions[JK[0], :] - Positions[i, :]
                    else:
                        v1 = Positions[i, :] - Positions[JK[0], :]

                    if SW_Fit[JK[1]] < SW_Fit[JK[2]]:  # Eq. (24)
                        v2 = Positions[JK[1], :] - Positions[JK[2], :]
                    else:
                        v2 = Positions[JK[2], :] - Positions[JK[1], :]

                    rn1 = np.random.randn()  # Normal distribution-based number
                    rn2 = np.random.randn()  # Normal distribution-based number

                    for j in range(dim):
                        SW_m[j] = Positions[i, j] + (np.exp(l)) * np.abs(rn1) * v1[j] + (1 - np.exp(l)) * np.abs(rn2) * v2[
                            j]  # Eq. (22)
                        if np.random.rand() < Cr:  # Eq. (21)
                            Positions[i, j] = SW_m[j]

                    # Return the search agents that exceed the search space's bounds
                    for j in range(dim):
                        if Positions[i, j] > ub[j]:
                            Positions[i, j] = lb[j] + np.random.rand() * (ub[j] - lb[j])
                        elif Positions[i, j] < lb[j]:
                            Positions[i, j] = lb[j] + np.random.rand() * (ub[j] - lb[j])

                    SW_Fit1 = feval(model_param, Positions[i, :])  # The fitness value of the newly generated spider
                    # Memory Saving and Updating the best-so-far solution
                    if SW_Fit1 < SW_Fit[i]:  # Change this to > for maximization problem
                        SW_Fit[i] = SW_Fit1  # Update the local best fitness
                        # Update the best-so-far solution
                        if SW_Fit[i] < Best_score:  # Change this to > for maximization problem
                            Best_score = SW_Fit[i]  # Update best-so-far fitness
                            Best_SW = Positions[i, :]  # Update best-so-far position
                    else:
                        Positions[i, :] = O_P  # Return the last best solution obtained by the ith solution

                    t += 1
                    if t >= Tmax:
                        break

                    Convergence_curve[t] = Best_score

            # Update the progress bar
            pbar.update(t - pbar.n)

        # Population reduction
        search_agents_no = int(N_min + (search_agents_no - N_min) * ((Tmax - t) / Tmax))  # Eq. (25)

    Convergence_curve[t - 1] = Best_score
    return Best_score, Best_SW, Convergence_curve
