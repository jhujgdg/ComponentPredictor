import torch
import pandas as pd
from Dataprocessing import get_data, load_data
from criterion import BinaryCrossEntropyLoss, calculate_accuracy
from old_model import TemporalConvNet

f1name = "component_bussion.xlsx"
# 加载数据
train_input, test_input, train_labels, test_labels, num_embeddings, y_train = load_data()

weight = torch.ones(num_embeddings)
embedding_dim = 128  # 词嵌入维度
num_channels = [16, 32, 64]
my_model = TemporalConvNet(num_embeddings, embedding_dim, num_channels)
optimizer = torch.optim.Adam(my_model.parameters(), lr=0.0004, weight_decay=0.001)
bce_loss = BinaryCrossEntropyLoss()

# 记录损失和准确率
results = []

for i in range(500):
    my_model.train()
    optimizer.zero_grad()
    outputs = my_model(train_input)
    loss = bce_loss.compute_loss(outputs, train_labels)

    # 反向传播
    loss.backward()
    optimizer.step()
    train_acc, train_precision, train_recall, train_F1 = calculate_accuracy(train_labels, outputs)

    my_model.eval()
    # 测试集评估
    test_outputs = my_model(test_input)
    test_loss = bce_loss.compute_loss(test_outputs, test_labels)
    test_acc, test_precision, test_recall, test_F1 = calculate_accuracy(test_labels, test_outputs)

    # 存储结果
    results.append({
        'iteration': i + 1,
        'train_loss': loss.item(),
        'test_loss': test_loss.item(),
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'train_precision': train_precision,
        'test_precision': test_precision,
        'train_recall': train_recall,
        'test_recall': test_recall,
        'train_F1': train_F1,
        'test_F1': test_F1
    })

    print(f'迭代次数: {i + 1}, train_loss: {loss.item()}, train_precision: {train_precision}, '
          f'train_recall: {train_recall}, train_F1: {train_F1}')
    print(f'       {i + 1}, test_loss: {test_loss.item()}, test_precision: {test_precision}, '
          f'test_recall: {test_recall}, test_F1: {test_F1}')
    print(f'迭代次数: {i + 1}, train_loss: {loss.item()}, train_acc: {train_acc}, '
          f'test_loss: {test_loss.item()}, test_acc: {test_acc}')

results_df = pd.DataFrame(results)
results_df.to_csv('TCN_results.csv', index=False)

