import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ======================
# 1. 构造数据
# ======================

chars = list("我他她它们的是在有和中大人民学习天气快乐生活中国北京上海广州深圳苹果香蕉猫狗山水火木金土")
special_char = "你"

chars = list(set(chars))  # 去重
vocab = chars + [special_char]

char2idx = {ch: i + 1 for i, ch in enumerate(vocab)}
vocab_size = max(char2idx.values()) + 1


def generate_sample():
    """
    生成一个长度为5的文本，其中包含一个'你'
    label = '你'所在的位置，范围 0~4
    """
    pos = random.randint(0, 4)
    text = []
    for i in range(5):
        if i == pos:
            text.append("你")
        else:
            text.append(random.choice(chars))
    label = pos
    return "".join(text), label


class PositionDataset(Dataset):
    def __init__(self, size=5000):
        self.data = [generate_sample() for _ in range(size)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label = self.data[idx]
        x = torch.tensor([char2idx[ch] for ch in text], dtype=torch.long)
        y = torch.tensor(label, dtype=torch.long)
        return x, y


train_dataset = PositionDataset(5000)
test_dataset = PositionDataset(1000)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)


# ======================
# 2. 定义模型
# ======================

class TextClassifier(nn.Module):
    def __init__(self, model_type="RNN", vocab_size=vocab_size, embed_dim=32, hidden_dim=64, num_classes=5):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        if model_type == "RNN":
            self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        elif model_type == "LSTM":
            self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        elif model_type == "GRU":
            self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        else:
            raise ValueError("model_type must be RNN, LSTM, or GRU")

        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        output, hidden = self.rnn(x)

        # 使用最后一个时间步的输出做分类
        last_output = output[:, -1, :]
        logits = self.fc(last_output)
        return logits


# ======================
# 3. 训练和测试函数
# ======================

def train_model(model, train_loader, test_loader, epochs=10, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        acc = evaluate(model, test_loader, device)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}, Test Acc: {acc:.4f}")

    return model


def evaluate(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            pred = torch.argmax(logits, dim=1)

            correct += (pred == y).sum().item()
            total += y.size(0)

    return correct / total


# ======================
# 4. 分别训练 RNN / LSTM / GRU
# ======================

for model_type in ["RNN", "LSTM", "GRU"]:
    print("=" * 40)
    print(f"Training {model_type}")
    model = TextClassifier(model_type=model_type)
    train_model(model, train_loader, test_loader, epochs=10)
