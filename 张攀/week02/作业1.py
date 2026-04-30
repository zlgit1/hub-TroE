import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'     # 修复plot无法画图的问题

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

"""

规律：
    输入5维向量，输出5维向量（最大的那个为1，其它为0）
    例如：
        输入：[0.22809727, 0.33051756, 0.04029306, 0.0612196 , 0.44784066]
        输出：[0., 0., 0., 0., 1.]

        输入：[0.67946254, 0.51620657, 0.39245569, 0.58503097, 0.16719904]
        输出：[1., 0., 0., 0., 0.]

"""


class TorchModel(nn.Module):
    def __init__(self, input_size=5, output_size=5):
        super().__init__()

        self.linear = nn.Linear(input_size, output_size)  # 获取类对象，形状变换
        # self.activation = torch.sigmoid         # 获取函数指针, 0~1分布，适合2分类或回归
        # self.activation = torch.softmax         # 概率分布，适合多分类
        
        # self.loss = nn.functional.mse_loss      # 获取函数指针
        self.loss = nn.CrossEntropyLoss()         # 获取类对象，输入直接为线性层输出（Logits），内部先softmax再计算交叉熵
        
    def forward(self, x, y=None):
        '''
        @x: 输入张量
        @y: 目标张量
        返回 5维预测张量 或 0维loss张量
        '''
        # x = self.linear(x) # 过线性层，n维 -> m维
        # y_pred = self.activation(x, 1) # 过非线性层，m维（形状不变，归一化）

        logits = self.linear(x) # 过线性层，batch * n维 -> batch * m维

        if y is not None:
            return self.loss(logits, y) # y * -lny^
        else:
            return torch.softmax(logits, 1)

def build_sample():
    '''
    返回 (5维输入向量, 5维目标向量)
    '''
    x = np.random.random(5)
    y = np.zeros(5)    
    i= np.argmax(x)     # 获取最大数的索引 np.where(x == np.max(x))[0]
    y[i] = 1
    return x, y

def build_dataset(total_sample_num):
    '''
    随机生成@total_sample_num数量训练集
    返回 (2阶输入张量, 2阶目标张量)
    '''
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    # print(X)
    # print(Y)
    # input()
    return torch.Tensor(X), torch.Tensor(Y)

def evaluate(model):
    '''
    模型测试
    '''
    model.eval()    # 进入evaluation 模式
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    print("进入评估模式：")
    correct, wrong = 0, 0
    with torch.no_grad():   # 不进行梯度计算
        # y_pred = model(x) # 模型预测 model.forward(x)
        y_pred = model.forward(x)

        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            pos_p = torch.argmax(y_p)
            pos_t = torch.argmax(y_t)
            if pos_p.item() == pos_t.item():
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    # 配置参数
    epoch_num = 300  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    output_size = 5 # 输入出向量维度
    learning_rate = 0.01  # 学习率

    # 建立模型
    model = TorchModel(input_size, output_size)

    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # optim = torch.optim.SGD(model.parameters(), lr=learning_rate)

    log = []
    # 创建训练集，正常任务是读取训练集
    train_x, train_y = build_dataset(train_sample)
    # 训练过程
    for epoch in range(epoch_num):
        model.train()       # 进入training 模式
        # model.eval()      # 进入evaluation 模式

        watch_loss = []
        for batch_index in range(train_sample // batch_size): # 不够一次batch 的数据舍弃
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]

            # loss = model(x, y)  # 计算loss  model.forward(x,y)
            loss = model.forward(x, y)
            
            loss.backward()  # 计算当前张量的梯度值
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零

            watch_loss.append(loss.item()) # 0阶张量转换为int/float

        print("=========\n第%d轮训练完成，平均loss:%f，样本数量:%d" % (epoch + 1, np.mean(watch_loss), len(watch_loss) * batch_size))
    
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "model.bin")
    # 画图
    # print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")     # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")    # 画loss曲线
    plt.legend()
    plt.show()
    return


def predict(model_name, input_vec):
    '''
    @model_name: 模型参数文件
    @input_vec: [[], [], ...]
    加载模型参数，根据input 打印one_hot形式预测结果
    '''
    input_size = 5
    output_size = 5
    model = TorchModel(input_size, output_size)
    model.load_state_dict(torch.load(model_name))  # 加载训练好的权重
    model_w1 = model.state_dict()["linear.weight"].numpy()
    model_b1 = model.state_dict()["linear.bias"].numpy()
    # print(model.state_dict())
    print(model_w1, "模型权重")
    print(model_b1, "模型偏置")
    print("=" * 40)

    model.eval()  # 进入测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.Tensor(input_vec))  # 模型预测

    for vec, res in zip(input_vec, result):
        # print(vec, res)
        idx = res.argmax()
        one_hot = nn.functional.one_hot(idx, num_classes=len(res))
        print(vec, one_hot)


if __name__ == "__main__":
    main()
    test_vec = [[0.88889086,0.15229675,0.31082123,0.03504317,0.88920843],
                [0.94963533,0.5524256,0.95758807,0.95520434,0.84890681],
                [0.90797868,0.67482528,0.13625847,0.34675372,0.19871392],
                [0.99349776,0.59416669,0.92579291,0.41567412,0.1358894]]
    predict("model.bin", test_vec)




