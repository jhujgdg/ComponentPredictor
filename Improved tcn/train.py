import torch
from Dataprocessing import get_data,load_data
from criterion import BinaryCrossEntropyLoss, calculate_accuracy
from model import TemporalConvNet
from visualization import Plotter

f1name = "component_bussion.xlsx"
# 加载数据
train_input, test_input, train_labels, test_labels, num_embeddings,y_train= load_data()

# weight = get_weight(y_train)
weight = torch.ones(num_embeddings)
embedding_dim = 390  # 词嵌入维度
num_channels = [16, 32, 64]
my_model = TemporalConvNet(num_embeddings, embedding_dim, num_channels,threshold=0.3382972160)
optimizer = torch.optim.Adam(my_model.parameters(), lr=0.0009993687, weight_decay=0.0001001703)
# criterion = get_criterion("BCELoss", num_embeddings, weight)
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
    test_outputs = my_model(test_input)  # 修改这一行以获取模型输出和注意力权重
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

    print('迭代次数:', i + 1, 'train_loss: ', loss.item(), 'train_precision: ',
          train_precision, 'train_recall: ', train_recall, 'train_F1: ', train_F1)
    print('迭代次数:', i + 1, 'test_loss：', test_loss.item(), 'test_precision: ', test_precision,
          'test_recall: ', test_recall, 'test_F1: ', test_F1)
    print('迭代次数:', i + 1, 'train_loss: ', loss.item(), 'train_acc: ',train_acc,'test_loss：', test_loss.item(), 'test_acc: ',test_acc)


# 可视化
p = Plotter(train_losses, test_losses, train_accse, test_accse,
            train_precisiones, test_precisiones, train_recalles, test_recalles)
p.plot()
# torch.save(my_model.state_dict(), "model1.pth")---BWOA
# torch.save(my_model.state_dict(), "model1.pth")---ABC
# torch.save(my_model.state_dict(), "model2.pth")---GOA
torch.save(my_model.state_dict(), "../joint model/Saved model/model3.pth")