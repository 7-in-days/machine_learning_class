import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt



# Input 넣고 output에 과연 어떤 이미지가 나올 것인가?
# latent space에서 Variation... 값들을 바꿔넣었을 때도 확인.
# 입력과 출력이 얼마나 잘 대응되는가?


# HyperParameters

batch_size = 128
latent_dim = 32
epochs = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   

# MNIST Load

transform = transforms.ToTensor() # 텐서 객체 생성
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# AutoEncoder 정의

class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        ) # 펼쳐서 linear -> Relu -> linear
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 28*28),
            nn.Sigmoid(), # 0~1 사이의 값으로 출력, 픽셀값 [0,1]
            nn.Unflatten(1, (1, 28, 28)) # 다시 이미지 형태로 변환
        )

        # decoder는 encoder의 반대 순서로 구성
        # encoder의 마지막 layer의 output이 decoder의 첫 layer의 input이 됨

    def forward(self, x):
        z = self.encoder(x) # latent space로 매핑
        out = self.decoder(z) # latent space에서 다시 이미지로 복원
        return out
    


# 학습

model = AutoEncoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 학습 루프

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for xb, _ in train_loader: # xb는 이미지 데이터, _는 레이블 (autoencoder에서는 레이블이 필요 없음)
        xb = xb.to(device)
        output = model(xb) 
        loss = criterion(output, xb) # 입력과 출력의 차이를 MSE로 계산

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()


    print(f'Epoch [{epoch+1}], Loss: {running_loss/len(train_loader):.4f}')


# 모델 평가

model.eval()
with torch.no_grad():
    images, _ = next(iter(test_loader)) # 테스트 데이터에서 한 배치 가져오기
    images = images.to(device) # 모델에 입력
    recon = model(images) # 모델의 출력 (latent space에서 복원된 이미지)


# 결과 시각화

n = 8
plt.figure(figsize=(16, 4))
for i in range(n):
    # 원본 이미지

    plt.subplot(2, n, i + 1)
    plt.imshow(images[i].cpu().squeeze(), cmap='gray')
    plt.axis("off") # 축 제거

    # 복원

    plt.subplot(2, n, n+i+1)
    plt.imshow(recon[i].cpu().squeeze(), cmap='gray')
    plt.axis("off")

plt.show()


# ex : latent_dim = 32일 때 [-3, 3] 사이의 벡터를 임의로 생성

with torch.no_grad():
    n = 8

    latent_vectors = torch.randn(n, latent_dim).to(device) # latent_dim 차원의 랜덤 벡터 생성

    # Decoder에 통과

    generated = model.decoder(latent_vectors) # 랜덤 벡터를 이미지로 복원

plt.figure(figsize=(16,2))

for i in range(n):
    plt.subplot(1, n, i + 1)
    plt.imshow(generated[i].cpu().squeeze(), cmap='gray')
    plt.axis("off")

plt.suptitle("랜덤 latent 벡터에서 생성된 이미지")
plt.show()



# # 테스트 이미지 하나 가져오기

# images, _ = next(iter(test_loader))
# x = images[0].unsqueeze(0).to(device) 

# # 인코딩된 latent vector

# with torch.no_grad():
#     z = model.encoder(x) # 입력 이미지를 latent space로 매핑


# # 특정 차원 변화시키기

# dim_to_modify = 0
# mod_range = torch.linspace(-3, 3, steps=7).to(device) # -3에서 3까지 8개의 값 생성

# # 수정된 latent vector로 이미지 생성

# modified_latents = z.repeat(len(mod_range), 1) # 원래 latent vector를 mod_range 길이만큼 복제
# modified_latents[:, dim_to_modify] += mod_range # 선택한 차원만 바꿈

# with torch.no_grad():
#     decoded_images = model.decoder(modified_latents)

# # 시각화

# plt.figure(figsize=(18, 2))
# for i in range(len(mod_range)):
#     plt.subplot(1, len(mod_range), i + 1)
#     plt.imshow(decoded_images[i].cpu().squeeze(), cmap='gray')
#     plt.axis("off")

# plt.suptitle(f"Latent dimension {dim_to_modify} 변화에 따른 이미지")
# plt.show()


