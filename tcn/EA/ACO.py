from tqdm import tqdm
import numpy as np
import torch
from tcn.model import TemporalConvNet
from tcn.criterion import BinaryCrossEntropyLoss, calculate_accuracy

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

def initialization(n_samples, n_var, ub, lb):
    # Initialize positions within the provided bounds for each dimension
    return np.random.uniform(lb, ub, (n_samples, n_var))


def ACO(aroa_param, model_param):
    n_pop = aroa_param['N']
    max_it = aroa_param['maxEvals']
    lb = aroa_param['lb']
    ub = aroa_param['ub']
    n_var = aroa_param['dim']

    # Problem Definition
    var_size = (1, n_var)  # Variables Matrix Size
    n_sample = 40  # Sample Size
    q = 0.5  # Intensification Factor (Selection Pressure)
    zeta = 1  # Deviation-Distance Ratio

    # Initialization
    pop = [{'Position': initialization(1, n_var, ub, lb)[0], 'Cost': None} for _ in range(n_pop)]

    for i in range(n_pop):
        pop[i]['Cost'] = cost_function(model_param, pop[i]['Position'])

    # Sort Population
    pop = sorted(pop, key=lambda x: x['Cost'])

    # Update Best Solution Ever Found
    best_sol = pop[0]
    best_cost = np.zeros(max_it)

    # Solution Weights
    w = 1 / (np.sqrt(2 * np.pi) * q * n_pop) * np.exp(-0.5 * (((np.arange(n_pop) - 1) / (q * n_pop)) ** 2))
    p = w / np.sum(w)

    # ACOR Main Loop with Progress Bar
    for it in tqdm(range(max_it), desc="Iterations", unit="iteration"):
        # Means
        s = np.array([ind['Position'] for ind in pop])

        # Standard Deviations
        sigma = np.zeros((n_pop, n_var))
        for l in range(n_pop):
            D = np.sum(np.abs(s[l] - s), axis=0)
            sigma[l] = zeta * D / (n_pop - 1)

        # Create New Population Array
        new_pop = []
        for t in range(n_sample):
            new_position = np.zeros(n_var)  # Corrected to 1D array of size n_var
            for i in range(n_var):
                l = roulette_wheel_selection(p)
                new_position[i] = s[l, i] + sigma[l, i] * np.random.randn()

                # Apply bounds for each dimension
                new_position[i] = np.clip(new_position[i], lb[i], ub[i])

            new_cost = cost_function(model_param, new_position)
            new_pop.append({'Position': new_position, 'Cost': new_cost})

        # Merge Main Population (Archive) and New Population (Samples)
        pop = pop + new_pop

        # Sort Population
        pop = sorted(pop, key=lambda x: x['Cost'])

        # Delete Extra Members
        pop = pop[:n_pop]

        # Update Best Solution Ever Found
        best_sol = pop[0]

        # Store Best Cost
        best_cost[it] = best_sol['Cost']

    # Results
    return best_cost[-1], best_sol['Position']


def roulette_wheel_selection(P):
    r = np.random.rand()
    C = np.cumsum(P)
    return np.where(r <= C)[0][0]
