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
    embedding_dim = int(x[0][1])
    num_channels = [16, 32, 64]
    my_model = TemporalConvNet(num_embeddings, embedding_dim, num_channels, threshold=x[0][3])
    optimizer = torch.optim.Adam(my_model.parameters(), lr=x[0][0], weight_decay=x[0][2])
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


def abc(aroa_param, model_param):
    n_pop = aroa_param['N']
    max_it = aroa_param['maxEvals']
    var_min = aroa_param['lb']
    var_max = aroa_param['ub']
    n_var = aroa_param['dim']

    # ABC Settings
    var_size = (1, n_var)  # Decision Variables Matrix Size
    n_onlooker = n_pop  # Number of Onlooker Bees
    l = round(0.6 * n_var * n_pop)  # Abandonment Limit Parameter (Trial Limit)
    a = 1  # Acceleration Coefficient Upper Bound

    # Initialization
    class Bee:
        def __init__(self):
            self.position = None
            self.cost = None

    # Initialize Population Array
    pop = [Bee() for _ in range(n_pop)]
    best_sol = Bee()
    best_sol.cost = np.inf

    # Create Initial Population
    for i in range(n_pop):
        pop[i].position = np.random.uniform(var_min, var_max, var_size)
        pop[i].cost = cost_function(model_param,pop[i].position)
        if pop[i].cost <= best_sol.cost:
            best_sol = pop[i]

    # Abandonment Counter
    c = np.zeros(n_pop)
    best_cost = np.zeros(max_it)

    # ABC Main Loop
    for it in tqdm(range(max_it), desc="Iterations"):
        # Recruited Bees
        for i in range(n_pop):
            # Choose k randomly, not equal to i
            k = np.random.choice(np.delete(np.arange(n_pop), i))
            phi = a * np.random.uniform(-1, 1, var_size)

            # New Bee Position
            newbee_position = pop[i].position + phi * (pop[i].position - pop[k].position)

            # Apply Bounds
            newbee_position = np.clip(newbee_position, var_min, var_max)

            # Evaluation
            newbee_cost = cost_function(model_param,newbee_position)

            # Comparison
            if newbee_cost <= pop[i].cost:
                pop[i].position = newbee_position
                pop[i].cost = newbee_cost
            else:
                c[i] += 1

        # Calculate Fitness Values and Selection Probabilities
        f = np.zeros(n_pop)
        mean_cost = np.mean([bee.cost for bee in pop])
        for i in range(n_pop):
            f[i] = np.exp(-pop[i].cost / mean_cost)  # Convert Cost to Fitness
        p = f / np.sum(f)

        # Onlooker Bees
        for m in range(n_onlooker):
            # Select Source Site
            i = roulette_wheel_selection(p)

            # Choose k randomly, not equal to i
            k = np.random.choice(np.delete(np.arange(n_pop), i))
            phi = a * np.random.uniform(-1, 1, var_size)

            # New Bee Position
            newbee_position = pop[i].position + phi * (pop[i].position - pop[k].position)

            # Apply Bounds
            newbee_position = np.clip(newbee_position, var_min, var_max)

            # Evaluation
            newbee_cost = cost_function(model_param,newbee_position)

            # Comparison
            if newbee_cost <= pop[i].cost:
                pop[i].position = newbee_position
                pop[i].cost = newbee_cost
            else:
                c[i] += 1

        # Scout Bees
        for i in range(n_pop):
            if c[i] >= l:
                pop[i].position = np.random.uniform(var_min, var_max, var_size)
                pop[i].cost = cost_function(model_param,pop[i].position)
                c[i] = 0

        # Update Best Solution Ever Found
        for i in range(n_pop):
            if pop[i].cost <= best_sol.cost:
                best_sol = pop[i]

        # Store Best Cost Ever Found
        best_cost[it] = best_sol.cost

    return best_sol.cost, best_sol.position, best_cost


def roulette_wheel_selection(p):
    r = np.random.rand()
    c = np.cumsum(p)
    return np.where(r <= c)[0][0]
