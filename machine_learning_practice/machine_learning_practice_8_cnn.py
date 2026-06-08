import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
import torch.nn.functional as F

transform = transforms.Compose([
    transforms.ToTensor(), # Tensor로 변환 
    transforms.Normalize((0.5, ), (0.5,)) # MNIST 평균, 표준편차로 정규화
])

# iteration 수 : 2000(trains) epoch 기준 2번?

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True) # 데이터 무작위로 섞기
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # Convolutional Layer 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Convolution layer 2

        # 채널 in, 채널 out, kernel_size(convolution filter의 크기), stride, padding size
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)


        # convolution layer 3

        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2) # 연산 후 128 * 2 * 2 = 512

        
        # fully connected layers

        self.fc1 = nn.Linear(512, 128) 
        self.fc2 = nn.Linear(128, 10) # output classes = 0~9


    def forward(self, x):
        
        # forward pass through conv layers

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        # flatten, fully connected layer

        x = x.view(-1, 512)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
    


# 모델 초기화

model = CNN()

# 손실함수, optimizer

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 학습

epochs = 3

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:

        # optimizer 초기화

        optimizer.zero_grad()

        # 모델의 순전파

        outputs = model(inputs)

        # 손실 계산

        loss = criterion(outputs, labels)

        loss.backward()

        # 가중치 업데이트

        optimizer.step()

        # 학습 상태 추적

        running_loss += loss.item()

        _, predicted = torch.max(outputs,1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f"Epoch [{epoch+1} / {epochs}], Loss : {running_loss/len(train_loader):.4f}, Accuracy : {100 * correct / total:.2f}%")


model.eval()
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total+= labels.size(0)
        correct+= (predicted == labels).sum().item()

print(f"Test Accuracy : {100 * correct / total:.2f}%")