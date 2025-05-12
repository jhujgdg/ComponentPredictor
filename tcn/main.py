import numpy as np
from EA import *
from Dataprocessing import load_data


def run_optimization(optimization_algorithm, algo_param, model_param):
    """
    执行指定的优化算法并返回结果

    :param optimization_algorithm: 具体的优化算法类或者函数
    :param algo_param: 算法相关的参数
    :param model_param: 模型相关的参数
    :return: best_err, best_learn_rate, best_cost
    """
    if optimization_algorithm == AROA:
        optimizer = optimization_algorithm(algo_param, model_param)
        return optimizer.run()
    elif optimization_algorithm == BWOA:
        return BWOA(algo_param, model_param)
    elif optimization_algorithm == GOA:
        return GOA(algo_param, model_param)
    elif optimization_algorithm == ACO:
        return ACO(algo_param, model_param)
    elif optimization_algorithm == GJO:
        return GJO(algo_param, model_param)
    elif optimization_algorithm == SWO:
        return SWO(algo_param, model_param)
    elif optimization_algorithm == SHO:
        return SHO(algo_param, model_param)
    elif optimization_algorithm == DMOA:
        return DMOA(algo_param, model_param)
    elif optimization_algorithm == MGO:
        return MGO(algo_param, model_param)
    else:
        raise ValueError("不支持的优化算法")


if __name__ == '__main__':
    filename = "loss和acc"
    # 生成训练集测试集
    train_input, test_input, train_labels, test_labels, num_embeddings, y_train = load_data()
    # 模型参数
    model_param = {
        "train_input": train_input,
        "test_input": test_input,
        "train_labels": train_labels,
        "test_labels": test_labels,
        "num_embeddings": num_embeddings,
        "y_train": y_train,
        "embedding_dim": 128,
        "hidden_size": 128,
        "num_layers": 1
    }

    ea_param = {
        "N": 10,
        "dim": 4,
        "lb": np.array([0.00001, 10, 0.0001, 0.01]),
        "ub": np.array([0.001, 400, 0.01, 1]),
        "maxEvals": 15
    }

    optimization_algorithm = MGO
    best_err, best_position, Convergence_curve = run_optimization(optimization_algorithm, ea_param, model_param)

    print(best_position)

    algo_name = optimization_algorithm.__name__

    with open('best_values.txt', 'a') as file:
        file.write("{}优化后最优F1为: {}\n".format(algo_name, best_err))
        file.write("优化后最优初始化参数: {}\n".format(best_position))

    print(best_position)
