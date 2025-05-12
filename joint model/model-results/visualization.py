import pandas as pd
import matplotlib.pyplot as plt

def plot_loss(filenames):
    # 训练集损失图
    plt.figure(figsize=(10, 6))
    for filename in filenames:
        data = pd.read_csv(filename)
        label = filename.split('_results.csv')[0]  # 获取文件名的前半部分
        plt.plot(data['iteration'], data['train_loss'], label=f'Train Loss - {label}', alpha=0.7)

    plt.title('Train Loss Comparison')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.show()

    # 测试集损失图
    plt.figure(figsize=(10, 6))
    for filename in filenames:
        data = pd.read_csv(filename)
        label = filename.split('_results.csv')[0]  # 获取文件名的前半部分
        plt.plot(data['iteration'], data['test_loss'], label=f'Test Loss - {label}', alpha=0.7)

    plt.title('Test Loss Comparison')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.show()


def plot_accuracy(filenames):
    # 训练集准确率图
    plt.figure(figsize=(10, 6))
    for filename in filenames:
        data = pd.read_csv(filename)
        label = filename.split('_results.csv')[0]  # 获取文件名的前半部分
        plt.plot(data['iteration'], data['train_accuracy'], label=f'Train Accuracy - {label}', alpha=0.7)

    plt.title('Train Accuracy Comparison')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    plt.show()

    # 测试集准确率图
    plt.figure(figsize=(10, 6))
    for filename in filenames:
        data = pd.read_csv(filename)
        label = filename.split('_results.csv')[0]  # 获取文件名的前半部分
        plt.plot(data['iteration'], data['test_accuracy'], label=f'Test Accuracy - {label}', alpha=0.7)

    plt.title('Test Accuracy Comparison')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    plt.show()


def plot_precision(filenames):
    # 训练集精确率图
    plt.figure(figsize=(10, 6))
    for filename in filenames:
        data = pd.read_csv(filename)
        label = filename.split('_results.csv')[0]  # 获取文件名的前半部分
        plt.plot(data['iteration'], data['train_precision'], label=f'Train Precision - {label}', alpha=0.7)

    plt.title('Train Precision Comparison')
    plt.xlabel('Iterations')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid()
    plt.show()

    # 测试集精确率图
    plt.figure(figsize=(10, 6))
    for filename in filenames:
        data = pd.read_csv(filename)
        label = filename.split('_results.csv')[0]  # 获取文件名的前半部分
        plt.plot(data['iteration'], data['test_precision'], label=f'Test Precision - {label}', alpha=0.7)

    plt.title('Test Precision Comparison')
    plt.xlabel('Iterations')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid()
    plt.show()


def plot_recall(filenames):
    # 训练集召回率图
    plt.figure(figsize=(10, 6))
    for filename in filenames:
        data = pd.read_csv(filename)
        label = filename.split('_results.csv')[0]  # 获取文件名的前半部分
        plt.plot(data['iteration'], data['train_recall'], label=f'Train Recall - {label}', alpha=0.7)

    plt.title('Train Recall Comparison')
    plt.xlabel('Iterations')
    plt.ylabel('Recall')
    plt.legend()
    plt.grid()
    plt.show()

    # 测试集召回率图
    plt.figure(figsize=(10, 6))
    for filename in filenames:
        data = pd.read_csv(filename)
        label = filename.split('_results.csv')[0]  # 获取文件名的前半部分
        plt.plot(data['iteration'], data['test_recall'], label=f'Test Recall - {label}', alpha=0.7)

    plt.title('Test Recall Comparison')
    plt.xlabel('Iterations')
    plt.ylabel('Recall')
    plt.legend()
    plt.grid()
    plt.show()


def plot_f1_score(filenames):
    # 训练集F1分数图
    plt.figure(figsize=(10, 6))
    for filename in filenames:
        data = pd.read_csv(filename)
        label = filename.split('_results.csv')[0]  # 获取文件名的前半部分
        plt.plot(data['iteration'], data['train_F1'], label=f'Train F1 - {label}', alpha=0.7)

    plt.title('Train F1 Score Comparison')
    plt.xlabel('Iterations')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid()
    plt.show()

    # 测试集F1分数图
    plt.figure(figsize=(10, 6))
    for filename in filenames:
        data = pd.read_csv(filename)
        label = filename.split('_results.csv')[0]  # 获取文件名的前半部分
        plt.plot(data['iteration'], data['test_F1'], label=f'Test F1 - {label}', alpha=0.7)

    plt.title('Test F1 Score Comparison')
    plt.xlabel('Iterations')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid()
    plt.show()


# 使用示例
filenames = ['BILSTM_results.csv', 'LSTM_results.csv',
             'RNN_results.csv', 'TCN_results.csv',]

plot_loss(filenames)
plot_accuracy(filenames)
plot_precision(filenames)
plot_recall(filenames)
plot_f1_score(filenames)
