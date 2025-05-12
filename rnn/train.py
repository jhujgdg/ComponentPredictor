import pandas as pd
import torch
from Dataprocessing import get_data, load_data
from criterion import BinaryCrossEntropyLoss, calculate_accuracy
from model import RNNNet
from visualization import Plotter

f1name = "component_bussion.xlsx"
train_input, test_input, train_labels, test_labels, num_embeddings, y_trin = load_data()

embedding_dim = 128
hidden_size = 128
num_layers = 2

my_model = RNNNet(input_size=num_embeddings, hidden_size=num_embeddings, output_size=num_embeddings, num_layers=2)
optimizer = torch.optim.Adam(my_model.parameters(), lr=0.00005, weight_decay=0.001)
bce_loss = BinaryCrossEntropyLoss()


# 记录损失和准确率
train_losses = []
test_losses = []
train_accse = []
test_accse = []
train_precisiones = []
test_precisiones = []
train_recalles = []
test_recalles = []

# 用于保存每次迭代的评估数据
iteration_data = []

for i in range(500):
    my_model.train()
    optimizer.zero_grad()  # 清零梯度
    outputs = my_model(train_input)
    loss = bce_loss.compute_loss(outputs, train_labels)

    # 反向传播
    loss.backward()
    optimizer.step()
    train_acc, train_precision, train_recall, train_F1 = calculate_accuracy(train_labels, outputs)

    my_model.eval()
    # 测试集评估
    test_outputs = my_model(test_input)  # 获取模型输出
    test_loss = bce_loss.compute_loss(test_outputs, test_labels)
    test_acc, test_precision, test_recall, test_F1 = calculate_accuracy(test_labels, test_outputs)

    # 存储损失和准确率
    train_losses.append(loss.item())
    test_losses.append(test_loss.item())
    train_precisiones.append(train_precision)
    test_precisiones.append(test_precision)
    train_recalles.append(train_recall)
    test_recalles.append(test_recall)
    train_accse.append(train_acc)
    test_accse.append(test_acc)

    # 保存每次迭代的数据
    iteration_data.append({
        'iteration': i + 1,
        'train_loss': loss.item(),
        'train_precision': train_precision,
        'train_recall': train_recall,
        'train_F1': train_F1,
        'test_loss': test_loss.item(),
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_F1': test_F1,
        'train_acc': train_acc,
        'test_acc': test_acc
    })

    print('迭代次数:', i + 1, 'train_loss: ', loss.item(), 'train_precision: ',
          train_precision, 'train_recall: ', train_recall, 'train_F1: ', train_F1)
    print('       ', i + 1, 'test_loss：', test_loss.item(), 'test_precision: ', test_precision,
          'test_recall: ', test_recall, 'test_F1: ', test_F1)
    print('迭代次数:', i + 1, 'train_loss: ', loss.item(), 'train_acc: ', train_acc,
          'test_loss：', test_loss.item(), 'test_acc: ', test_acc)

df = pd.DataFrame(iteration_data)
df.to_csv('RNN_results.csv', index=False)

# 可视化
p = Plotter(train_losses, test_losses, train_accse, test_accse,
            train_precisiones, test_precisiones, train_recalles, test_recalles)
p.plot()
