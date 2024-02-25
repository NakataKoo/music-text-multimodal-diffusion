import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# データセットの前処理とロード
transform = transforms.Compose([
    transforms.ToTensor(),
    # 他の前処理ステップがあればここに追加
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# モデルの定義
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        # モデルのアーキテクチャを定義
        self.layer1 = nn.Linear(28*28, 64)
        self.layer2 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # フラット化
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
        return x

model = NeuralNet()

# 損失関数とオプティマイザ
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 学習ループ
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (data, targets) in enumerate(train_loader):
        # フォワードパス
        outputs = model(data)
        loss = criterion(outputs, targets)

        # バックプロパゲーション
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # エポックの終了ごとにログを出力
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# モデルの評価
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for data, targets in test_loader:
        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

accuracy = 100 * correct / total
print(f'Accuracy of the model on the test images: {accuracy:.2f}%')
