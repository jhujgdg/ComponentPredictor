import matplotlib.pyplot as plt


class Plotter:
    def __init__(self, train_losses, test_losses, train_acc, test_acc,
                 train_precisions, test_precisions, train_recalls, test_recalls):
        self.train_losses = train_losses
        self.test_losses = test_losses
        self.train_acc = train_acc
        self.test_acc = test_acc
        self.train_precisions = train_precisions
        self.test_precisions = test_precisions
        self.train_recalls = train_recalls
        self.test_recalls = test_recalls

    def plot_loss(self):
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(self.train_losses) + 1), self.train_losses, label='Train Loss', marker='o', markersize=3)
        plt.plot(range(1, len(self.test_losses) + 1), self.test_losses, label='Test Loss', marker='o', markersize=3)
        plt.title('Loss vs Iterations')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid()
        plt.show()

    def plot_accuracy(self):
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(self.train_acc) + 1), self.train_acc, label='Train Accuracy', marker='o', markersize=3)
        plt.plot(range(1, len(self.test_acc) + 1), self.test_acc, label='Test Accuracy', marker='o', markersize=3)
        plt.title('Accuracy vs Iterations')
        plt.xlabel('Iterations')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)  # 准确率范围
        plt.legend()
        plt.grid()
        plt.show()

    def plot_precision_recall(self):
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(self.train_precisions) + 1), self.train_precisions, label='Train Precision', marker='o',
                 markersize=3)
        plt.plot(range(1, len(self.test_precisions) + 1), self.test_precisions, label='Test Precision', marker='o',
                 markersize=3)
        plt.plot(range(1, len(self.train_recalls) + 1), self.train_recalls, label='Train Recall', marker='o',
                 markersize=3)
        plt.plot(range(1, len(self.test_recalls) + 1), self.test_recalls, label='Test Recall', marker='o', markersize=3)
        plt.title('Precision and Recall vs Iterations')
        plt.xlabel('Iterations')
        plt.ylabel('Value')
        plt.ylim(0, 1)  # 精确率和召回率范围
        plt.legend()
        plt.grid()
        plt.show()

    def plot(self):
        self.plot_loss()
        self.plot_accuracy()
        self.plot_precision_recall()
