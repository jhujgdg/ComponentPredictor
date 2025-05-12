import pickle
from collections import defaultdict
import torch
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd


def get_all_labels(input_sequences, output_labels):
    all_labels = []
    # 遍历输入序列和标签,将标签添加到对应的列表中
    for i, seq in enumerate(input_sequences):
        labels = []
        labels.append(output_labels[i])
        for j, seq1 in enumerate(input_sequences):
            if i != j and (seq1 == seq).all():
                labels.append(output_labels[j])
        all_labels.append(labels)
    all_labels = [list(set(labels)) for labels in all_labels]
    return all_labels


def get_data(fname):
    output_df = pd.read_excel(fname, engine="openpyxl")

    all_component_ids = []

    # 遍历 output_df 中的每一行
    for index, row in output_df.iterrows():
        component_ids = row['构件ID']

        # 将构件 ID 字符串转换为列表,并添加到 all_component_ids 列表中
        component_ids_list = component_ids.split(',')
        # print(component_ids_list)
        all_component_ids.extend(component_ids_list)
    # 将 all_component_ids 列表转换为集合,自动去重
    sentences = list(set(all_component_ids))

    # 创建一个Tokenizer对象，并用数据集拟合它
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(sentences)

    # 将文本转换为序列
    sequences = tokenizer.texts_to_sequences(sentences)
    num_embeddings = len(tokenizer.word_index) + 1

    input_sequences = []
    output_labels = defaultdict(list)  # 使用 defaultdict 来收集标签

    # 生成 n-gram 序列
    for sequence in sequences:
        for i in range(1, len(sequence)):
            n_gram_sequence = sequence[:i + 1]
            input_seq = n_gram_sequence[:-1]
            label = n_gram_sequence[-1]

            # 添加到 input_sequences
            input_sequences.append(input_seq)

            # 将标签添加到对应的 input_seq 中
            output_labels[tuple(input_seq)].append(label)

    # 将 output_labels 转换为二维数组
    output_labels_list = []
    for input_seq in input_sequences:
        output_labels_list.append(output_labels[tuple(input_seq)])
    labels = []
    for label in output_labels_list:
        array = np.zeros(num_embeddings)
        for i in label:
            array[i - 1] = 1
        labels.append(array)
    # 对输入序列进行填充
    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
    # 创建一个列表,用于存储每个输入序列对应的标签列表
    # all_labels = get_all_labels(input_sequences,output_labels)
    # print(all_labels)
    input = torch.tensor(input_sequences)
    labels = torch.tensor(labels,dtype=torch.float32)

    X_train, X_test, y_train, y_test= train_test_split(input, labels, test_size=0.3, random_state=42)


    Y_train = pd.Series(y_train, name='label')

    return X_train, X_test, y_train, y_test, num_embeddings,Y_train


def load_data():
    data = torch.load('data.pth1')
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    num_embeddings = data['num_embeddings']
    Y_train = data['Y_train']
    return X_train, X_test, y_train, y_test, num_embeddings, Y_train


if __name__ == '__main__':
    f1name = "component_bussion.xlsx"
    get_data(f1name)

