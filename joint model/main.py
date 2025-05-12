import numpy as np
from AROA import AROA
from Dataprocessing import load_data

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

    aroa_param = {
        "N": 100,
        "dim": 4,
        "lb": 0,
        "ub": 1,
        "maxEvals": 1000
    }
    aroa = AROA(aroa_param, model_param)
    best_err, best_learn_rate = aroa.run()

    print(best_learn_rate)
    with open('best_values.txt', 'a') as file:
        file.write("人类进化优化后最优F1为: {}\n".format(best_err))
        file.write("优化后最优初始化参数: {}\n".format(best_learn_rate))
    print(best_learn_rate)

