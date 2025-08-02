import numpy as np
from scipy.special import gamma
from tqdm import tqdm
import torch
from pygcn import accuracy
from ..model import TemporalConvNet
from tcn.criterion import BinaryCrossEntropyLoss, calculate_accuracy


def init(SearchAgents, dimension, upperbound, lowerbound):
    Boundary = upperbound.shape[0]
    if Boundary == 1:
        Pos = np.random.rand(SearchAgents, dimension) * (upperbound - lowerbound) + lowerbound
    else:
        Pos = np.zeros((SearchAgents, dimension))
        for i in range(dimension):
            ub_i = upperbound[i]
            lb_i = lowerbound[i]
            Pos[:, i] = np.random.rand(SearchAgents) * (ub_i - lb_i) + lb_i
    return Pos


def noh(best_hyena_fitness):
    min_val = 0.5
    max_val = 1
    count = 0
    M = (max_val - min_val) * np.random.rand() + min_val + best_hyena_fitness[0]

    for i in range(1, len(best_hyena_fitness)):
        if M >= best_hyena_fitness[i]:
            count += 1

    return count


def SHO(aroa_param, model_param):
    N = aroa_param['N']
    Max_iterations = aroa_param['maxEvals']
    lowerbound = aroa_param['lb']
    upperbound = aroa_param['ub']
    dimension = aroa_param['dim']

    hyena_pos = init(N, dimension, upperbound, lowerbound)
    Convergence_curve = np.zeros(Max_iterations)

    Iteration = 0

    # 在循环之前设置进度条
    with tqdm(total=Max_iterations, desc="Iterations Progress") as pbar:
        while Iteration < Max_iterations:
            hyena_fitness = np.zeros(N)

            for i in range(N):
                # Boundary check
                H_ub = hyena_pos[i, :] > upperbound
                H_lb = hyena_pos[i, :] < lowerbound
                hyena_pos[i, :] = (hyena_pos[i, :] * ~(H_ub | H_lb)) + upperbound * H_ub + lowerbound * H_lb
                hyena_fitness[i] = cost_function(model_param, hyena_pos[i, :])

            if Iteration == 0:
                fitness_sorted = np.sort(hyena_fitness)
                FS = np.argsort(hyena_fitness)
                sorted_population = hyena_pos[FS, :]
                best_hyenas = sorted_population
                best_hyena_fitness = fitness_sorted
            else:
                double_population = np.vstack((pre_population, best_hyenas))
                double_fitness = np.hstack((pre_fitness, best_hyena_fitness))
                double_fitness_sorted = np.sort(double_fitness)
                FS = np.argsort(double_fitness)
                double_sorted_population = double_population[FS, :]
                fitness_sorted = double_fitness_sorted[:N]
                sorted_population = double_sorted_population[:N, :]
                best_hyenas = sorted_population
                best_hyena_fitness = fitness_sorted

            NOH = noh(best_hyena_fitness)

            Best_hyena_score = fitness_sorted[0]
            Best_hyena_pos = sorted_population[0, :]
            pre_population = hyena_pos.copy()
            pre_fitness = hyena_fitness.copy()

            a = 5 - Iteration * (5 / Max_iterations)
            for i in range(N):
                CV = 0
                for j in range(dimension):
                    for k in range(NOH):  # Corrected from NOHS to NOH
                        r1 = np.random.rand()
                        r2 = np.random.rand()
                        Var1 = 2 * a * r1 - a
                        Var2 = 2 * r2
                        distance_to_hyena = abs(Var2 * sorted_population[k, j] - hyena_pos[i, j])
                        HYE = sorted_population[k, j] - Var1 * distance_to_hyena
                        CV += HYE

                    hyena_pos[i, j] = CV / (NOH + 1)  # Ensure this is a scalar value

            Convergence_curve[Iteration] = Best_hyena_score
            Iteration += 1

            # 更新进度条
            pbar.update(1)  # 更新进度条

    return Best_hyena_score, Best_hyena_pos, Convergence_curve


def cost_function(model_param, x):
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
    # criterion = get_criterion("BCELoss", num_embeddings, weight)
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
        test_outputs = my_model(test_input)  # 修改这一行以获取模型输出和注意力权重
        test_loss = bce_loss.compute_loss(test_outputs, test_labels)
        test_acc, test_precision, test_recall, test_F1 = calculate_accuracy(test_labels, test_outputs)

        history.append((-1 * test_F1))

    return min(history)
