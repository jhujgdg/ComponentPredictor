import torch
import model
from Dataprocessing import load_data
from criterion import calculate_accuracy, BinaryCrossEntropyLoss

x = [0.02305564, 0.1287994, 0.0786522, 0.25158903]
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

model1.eval()
model2.eval()
model3.eval()
model4.eval()

with torch.no_grad():
    output1 = model1(test_input)
    output2 = model2(test_input)
    output3 = model3(test_input)
    output4 = model4(test_input)
# 加权结合
bce_loss = BinaryCrossEntropyLoss()
output = (x[0] * output1 + x[1] * output2 + x[2] * output3 + x[3] * output4) / (x[0] + x[1] + x[2] + x[3])
test_loss = bce_loss.compute_loss(output, test_labels)
test_acc, test_precision, test_recall, test_F1 = calculate_accuracy(test_labels, output)
print(test_loss,test_acc, test_precision, test_recall, test_F1)
