import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 1. 데이터 생성: 0.5에서 시작해 ±0.05씩 움직이는 1D Periodic Random Walk
def generate_agent_data(n_steps=1000, start_pos=0.5, step_size=0.05):
    path = [start_pos]
    for _ in range(n_steps - 1):
        # ±step_size 범위 내에서 무작위 이동
        move = np.random.uniform(-step_size, step_size)
        next_pos = (path[-1] + move) % 1.0  # 1.0을 넘으면 0으로 돌아오는 순환 구조
        path.append(next_pos)
    return np.array(path).reshape(-1, 1)

# 데이터 준비
n_steps = 1500
raw_samples = generate_agent_data(n_steps=n_steps)
data_tensor = torch.FloatTensor(raw_samples)

# 2. Autoencoder 모델 정의
class SensoryAE(nn.Module):
    def __init__(self):
        super(SensoryAE, self).__init__()
        # Encoder: 1D 입력을 2D 잠재 공간으로 확장 (원형 구조를 표현하기 위해 2D가 유리함)
        self.encoder = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 2) 
        )
        # Decoder: 2D 잠재 공간에서 다시 1D로 복원
        self.decoder = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid() # 0~1 사이 값 복원을 위해 Sigmoid 사용
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

model = SensoryAE()
optimizer = optim.Adam(model.parameters(), lr=0.005)
criterion = nn.MSELoss()

# 3. 모델 학습
epochs = 200
losses = []

for epoch in range(epochs):
    optimizer.zero_grad()
    x_hat, z = model(data_tensor)
    loss = criterion(x_hat, data_tensor)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    
    if (epoch+1) % 50 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")

# 4. 결과 시각화
with torch.no_grad():
    reconstructed, latent = model(data_tensor)

plt.figure(figsize=(15, 5))

# (1) 원본 데이터 흐름
plt.subplot(1, 3, 1)
plt.plot(raw_samples[:100], 'b-', alpha=0.6)
plt.title("Sensory Data Stream (First 100 steps)")
plt.xlabel("Time Step")
plt.ylabel("Position (0-1)")

# (2) 잠재 공간 (Latent Space) - AE가 그린 내부 지도
# 0~1 값에 따라 색상을 입혀 위상적 연속성을 확인
plt.subplot(1, 3, 2)
plt.scatter(latent[:, 0], latent[:, 1], c=raw_samples.flatten(), cmap='hsv', s=10)
plt.title("Latent Space (Internal Map)")
plt.xlabel("z1")
plt.ylabel("z2")
plt.colorbar(label='Original Position')

# (3) 복원 결과 비교
plt.subplot(1, 3, 3)
plt.scatter(raw_samples, reconstructed, alpha=0.5, s=10)
plt.plot([0, 1], [0, 1], 'r--') # 정답선
plt.title("Original vs Reconstructed")
plt.xlabel("Original")
plt.ylabel("Reconstructed")

plt.tight_layout()
plt.show()
