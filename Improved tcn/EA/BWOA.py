from tqdm import tqdm
import numpy as np
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

def initialization(search_agents_no, dim, ub, lb):
    # 使用不同的上下界初始化每个维度
    return np.array([np.random.uniform(lb[i], ub[i], search_agents_no) for i in range(dim)]).T


def get_pheromone(fit, min_fit, max_fit):
    return (max_fit - fit) / (max_fit - min_fit)


def get_binary():
    return np.random.randint(0, 2)


def BWOA(aroa_param, model_param):
    search_agents_no = aroa_param['N']
    max_iter = aroa_param['maxEvals']
    lb = aroa_param['lb']
    ub = aroa_param['ub']
    dim = aroa_param['dim']
    positions = initialization(search_agents_no, dim, ub, lb)
    fitness = np.array([fobj(model_param, pos) for pos in positions])

    v_min = np.min(fitness)
    min_idx = np.argmin(fitness)
    the_best_vct = positions[min_idx, :]

    v_max = np.max(fitness)
    convergence_curve = np.zeros(max_iter + 1)  # Increased size to max_iter + 1
    convergence_curve[0] = v_min
    pheromone = get_pheromone(fitness, v_min, v_max)

    # Main loop with tqdm progress bar
    for t in tqdm(range(max_iter), desc="Iterations"):
        beta = -1 + 2 * np.random.rand()  # Random value between -1 and 1
        m = 0.4 + 0.5 * np.random.rand()  # Random value between 0.4 and 0.9

        for r in range(search_agents_no):
            P = np.random.rand()
            r1 = np.random.randint(0, search_agents_no)

            if P >= 0.3:  # Spiral search
                v = the_best_vct - np.cos(2 * np.pi * beta) * positions[r, :]
            else:  # Direct search
                v = the_best_vct - m * positions[r1, :]

            if pheromone[r] <= 0.3:
                band = True
                while band:
                    r1 = np.random.randint(0, search_agents_no)
                    r2 = np.random.randint(0, search_agents_no)
                    if r1 != r2:
                        band = False
                v = the_best_vct + (positions[r1, :] - (-1) ** get_binary() * positions[r2, :]) / 2

            # Return back the search agents that go beyond the boundaries of the search space
            # 使用相应维度的上下界进行裁剪
            v = np.clip(v, lb, ub)

            # Evaluate new solutions
            f_new = fobj(model_param, v)

            # Update if the solution improves
            if f_new <= fitness[r]:
                positions[r, :] = v
                fitness[r] = f_new

            if f_new <= v_min:
                the_best_vct = v
                v_min = f_new

        # Update max and pheromones
        v_max = np.max(fitness)
        pheromone = get_pheromone(fitness, v_min, v_max)
        convergence_curve[t + 1] = v_min  # This is now safe

    return v_min, the_best_vct, convergence_curve