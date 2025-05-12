from tqdm import tqdm
import numpy as np
import torch
import model
from Dataprocessing import load_data
from criterion import calculate_accuracy


def fobj(model_param, x):
    train_input, test_input, train_labels, test_labels, num_embeddings, y_train = load_data()

    num_channels = [16, 32, 64]
    model1 = model.TemporalConvNet(num_embeddings, embedding_dim=261, num_channels=num_channels, threshold=0.0330378288)
    model2 = model.TemporalConvNet(num_embeddings, embedding_dim=400, num_channels=num_channels, threshold=0.0665064361)
    model3 = model.TemporalConvNet(num_embeddings, embedding_dim=397, num_channels=num_channels, threshold=0.4106046590)
    model4 = model.TemporalConvNet(num_embeddings, embedding_dim=390, num_channels=num_channels, threshold=0.3382972160)

    # 加载模型权重
    model1.load_state_dict(torch.load('Saved model/model.pth'))
    model2.load_state_dict(torch.load('Saved model/model1.pth'))
    model3.load_state_dict(torch.load('Saved model/model2.pth'))
    model4.load_state_dict(torch.load('Saved model/model3.pth'))

    # 设置模型为评估模式
    model1.eval()
    model2.eval()
    model3.eval()
    model4.eval()

    # 使用测试输入进行预测
    with torch.no_grad():  # 在推理时关闭梯度计算
        output1 = model1(test_input)
        output2 = model2(test_input)
        output3 = model3(test_input)
        output4 = model4(test_input)
    # 加权结合
    output = (x[0] * output1 + x[1] * output2 + x[2] * output3 + x[3] * output4) / (x[0] + x[1] + x[2] + x[3])
    test_acc, test_precision, test_recall, test_F1 = calculate_accuracy(test_labels, output)
    cost = -1*test_F1
    return cost




class AROA:
    def __init__(self, aroa_param, model_param):
        self.N = aroa_param['N']
        self.lb = aroa_param['lb']
        self.ub = aroa_param['ub']
        self.maxEvals = aroa_param['maxEvals']
        self.dim = aroa_param['dim']
        self.model_param = model_param

        self.c = 0.95
        self.fr1 = 0.15
        self.fr2 = 0.6
        self.p1 = 0.2
        self.p2 = 0.8
        self.Ef = 0.4
        self.tr1 = 0.9
        self.tr2 = 0.85
        self.tr3 = 0.9
        self.tmax = int(np.ceil((self.maxEvals - self.N) / (2 * self.N)))  # 最大迭代次数
        self.evalCounter = 0  # 初始化一个计数器evalCounter，用于跟踪到目前为止已经进行的评估次数

        self.Convergence_curve = np.zeros(self.tmax)  # 记录每次迭代后的最佳适应度。数组的长度设置为tmax，初始值为0
        self.Xmin = np.tile(np.ones(self.dim) * self.lb, (self.N, 1))  # 定义边界矩阵
        self.Xmax = np.tile(np.ones(self.dim) * self.ub, (self.N, 1))

        self.X = np.random.rand(self.N, self.dim) * (
                    self.ub - self.lb) + self.lb  # 从区间 [lb, ub] 内随机生成 N 个个体，每个个体的每个维度值都在 [lb, ub] 范围内
        self.X, self.F, self.evalCounter = self.evaluate_population(self.X, self.ub, self.lb, self.evalCounter,
                                                                    self.maxEvals)
        self.fbest, self.ibest = np.min(self.F), np.argmin(self.F)  # 当前种群中目标函数值最小的个体及其索引
        self.xbest = self.X[self.ibest, :]  # 目标函数最小的所有信息
        self.X_memory = self.X.copy()
        self.F_memory = self.F.copy()

    def run(self):
        for t in tqdm(range(1, int(self.tmax) + 1), desc="Iterations"):
            # 计算每个个体之间的距离平方
            D = np.sum((self.X[:, np.newaxis, :] - self.X) ** 2, axis=2)
            m = self.tanh(t, self.tmax, [-2, 7])
            for i in range(self.N):
                Dimax = np.max(D[i, :])  # 个体 i 的邻居中最大的距离平方
                k = int(np.floor((1 - t / self.tmax) * self.N) + 1)
                neighbors = np.argsort(D[i, :])  # 根据距离排序个体 i 的邻居

                delta_ni = np.zeros(self.dim)
                for j in neighbors:
                    I = 1 - (D[i, j] / Dimax)
                    s = np.sign(self.F[j] - self.F[i])
                    delta_ni += self.c * (self.X_memory[i] - self.X_memory[j]) * I * s
                ni = delta_ni / self.N

                if np.random.rand() < self.p1:
                    bi = m * self.c * (np.random.rand(self.dim) * self.xbest - self.X_memory[i])
                else:
                    bi = m * self.c * (self.xbest - self.X_memory[i])

                if np.random.rand() < self.p2:
                    if np.random.rand() > 0.5 * t / self.tmax + 0.25:
                        u1 = np.random.rand(self.dim) > self.tr1
                        ri = u1 * np.random.normal(0, self.fr1 * (1 - t / self.tmax) * (self.ub - self.lb), self.dim)
                    else:
                        u2 = np.random.rand(self.dim) > self.tr2
                        w = self.index_roulette_wheel_selection(self.F, k)
                        Xw = self.X_memory[w]
                        if np.random.rand() < 0.5:
                            ri = self.fr2 * u2 * (1 - t / self.tmax) * np.sin(
                                2 * np.pi * np.random.rand(self.dim)) * np.abs(
                                np.random.rand(self.dim) * Xw - self.X_memory[i])
                        else:
                            ri = self.fr2 * u2 * (1 - t / self.tmax) * np.cos(
                                2 * np.pi * np.random.rand(self.dim)) * np.abs(
                                np.random.rand(self.dim) * Xw - self.X_memory[i])
                else:
                    u3 = np.random.rand(self.dim) > self.tr3
                    ri = u3 * (2 * np.random.rand(self.dim) - np.ones(self.dim)) * (self.ub - self.lb)

                self.X[i] = self.X[i] + ni + bi + ri

            self.X, self.F, self.evalCounter = self.evaluate_population(self.X, self.ub, self.lb, self.evalCounter,
                                                                        self.maxEvals)
            fbest_candidate, ibest_candidate = np.min(self.F), np.argmin(self.F)

            if fbest_candidate < self.fbest:
                self.fbest = fbest_candidate
                self.xbest = self.X[ibest_candidate]

            self.X, self.F = self.memory_operator(self.X, self.F, self.X_memory, self.F_memory)
            self.X_memory = self.X.copy()
            self.F_memory = self.F.copy()

            CF = (1 - t / self.tmax) ** 3
            if np.random.rand() < self.Ef:
                u4 = np.random.rand(self.N, self.dim) < self.Ef
                self.X = self.X + CF * (u4 * (np.random.rand(self.N, self.dim) * (self.Xmax - self.Xmin) + self.Xmin))
            else:
                r7 = np.random.rand()
                self.X = self.X + (CF * (1 - r7) + r7) * (
                        self.X[np.random.permutation(self.N)] - self.X[np.random.permutation(self.N)])

            self.X, self.F, self.evalCounter = self.evaluate_population(self.X, self.ub, self.lb, self.evalCounter,
                                                                        self.maxEvals)
            fbest_candidate, ibest_candidate = np.min(self.F), np.argmin(self.F)

            if fbest_candidate < self.fbest:
                self.fbest = fbest_candidate
                self.xbest = self.X[ibest_candidate]

            self.X, self.F = self.memory_operator(self.X, self.F, self.X_memory, self.F_memory)

            self.Convergence_curve[t - 1] = self.fbest

        return self.fbest, self.xbest  # , self.Convergence_curve

    def evaluate_population(self, X, ub, lb, evalCounter,
                            maxEvals):  # 函数用于评估种群中每个个体的适应度值。在给定的搜索空间范围内，对每个个体的位置进行边界处理，并计算其适应度值。
        N = X.shape[0]
        F = np.full(N, np.inf)  # 初始化一个长度为 N 的数组 F，用于存储每个个体的适应度值。初始时，所有适应度值都被设置为无穷大，这表示还没有评估任何个体
        X = np.maximum(lb, np.minimum(ub, X))  # 确保每个个体的每个维度都在 [lb, ub] 范围内

        # 假设 N 是种群个体的数量，maxEvals 是最大评估次数
        for i in tqdm(range(N), desc="Evaluating individuals"):
            if evalCounter >= maxEvals:
                break
            F[i] = fobj(self.model_param, X[i])  # 对种群个体评分
            evalCounter += 1

        return X, F, evalCounter

    def memory_operator(self, X, F, X_memory, F_memory):  # 函数根据当前解与记忆解之间的适应度比较，更新种群中的个体位置和适应度值。
        Inx = F_memory < F
        Indx = np.tile(Inx[:, np.newaxis], (1, X.shape[1]))
        X = Indx * X_memory + ~Indx * X
        F = Inx * F_memory + ~Inx * F
        return X, F

    def tanh(self, t, tmax, range):
        z = 2 * (t / tmax * (range[1] - range[0]) + range[0])
        y = 0.5 * ((np.exp(z) - 1) / (np.exp(z) + 1) + 1)
        return y

    def index_roulette_wheel_selection(self, F, k):
        fitness = F[:k]
        # 计算权重，注意避免除以零
        weights = np.max(fitness) - fitness
        # 计算权重的累积和，注意检查分母是否为零
        weights_sum = np.sum(weights)
        if weights_sum == 0:
            # 如果分母为零，设置一个非零的默认值
            weights_sum = 1
        weights = np.cumsum(weights / weights_sum)

        return self.roulette_wheel_selection(weights)

    def roulette_wheel_selection(self, weights):
        r = np.random.rand()
        selected_index = 1
        for index, weight in enumerate(weights):
            if r <= weight:
                selected_index = index
                break
        return selected_index
