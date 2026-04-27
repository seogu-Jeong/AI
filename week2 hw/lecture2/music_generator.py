import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import random

# GPU 세팅
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"현재 엔진: {device}")

# 1. 데이터셋 준비 (Irish Folk Song Sample)
# mitdeeplearning 대신 샘플 데이터를 직접 포함합니다.
print("데이터셋 준비 중...")
sample_songs = [
    "X:1\nT:Alexander's\nZ: id:dc-hornpipe-1\nM:4/4\nL:1/8\nK:D Major\n(3ABc|d2dA F2AF|E2EG FDFA|d2dA F2AF|E2EG FDDA|!",
    "X:2\nT:All Around The World\nZ: id:dc-hornpipe-2\nM:4/4\nL:1/8\nK:G Major\n(3def|g2gf gdBd|cBce dBGB|A2AB AGEG|DGGF G2:|!",
    "X:3\nT:Bantry Bay\nZ: id:dc-hornpipe-3\nM:4/4\nL:1/8\nK:G Major\nga|bgag egde|gedB AGEG|D2GB AGED|G2GF G2:|!"
]
# 반복해서 데이터 양을 늘림
songs_joined = "\n\n".join(sample_songs * 100)
vocab = sorted(set(songs_joined))

char2idx = {u: i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

def vectorize_string(string):
    return np.array([char2idx[char] for char in string])

vectorized_songs = vectorize_string(songs_joined)

def get_batch(vectorized_songs, seq_length, batch_size):
    n = vectorized_songs.shape[0] - 1
    idx = np.random.choice(n - seq_length, batch_size)
    input_batch = [vectorized_songs[i : i + seq_length] for i in idx]
    output_batch = [vectorized_songs[i + 1 : i + seq_length + 1] for i in idx]
    return torch.tensor(input_batch, dtype=torch.long), torch.tensor(output_batch, dtype=torch.long)

# 2. RNN (LSTM) 모델 정의
class MusicGeneratorModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super(MusicGeneratorModel, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def init_hidden(self, batch_size, device):
        return (torch.zeros(1, batch_size, self.hidden_size).to(device),
                torch.zeros(1, batch_size, self.hidden_size).to(device))

    def forward(self, x, state=None, return_state=False):
        x = self.embedding(x)
        if state is None:
            state = self.init_hidden(x.size(0), x.device)
        out, state = self.lstm(x, state)
        out = self.fc(out)
        return out if not return_state else (out, state)

# 3. 하이퍼파라미터 및 훈련 세팅
vocab_size = len(vocab)
embedding_dim = 256
hidden_size = 512 # 로컬 실행을 위해 메모리 사용량 최적화
batch_size = 16
seq_length = 64
learning_rate = 5e-3
num_training_iterations = 500

model = MusicGeneratorModel(vocab_size, embedding_dim, hidden_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# 4. 모델 훈련
print("훈련 시작...")
model.train()
for iter in tqdm(range(num_training_iterations)):
    x_batch, y_batch = get_batch(vectorized_songs, seq_length, batch_size)
    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
    optimizer.zero_grad()
    y_hat = model(x_batch)
    loss = criterion(y_hat.view(-1, vocab_size), y_batch.view(-1))
    loss.backward()
    optimizer.step()

# 5. 작곡
def generate_text(model, start_string="X", generation_length=500):
    input_idx = [char2idx[s] for s in start_string]
    input_idx = torch.tensor([input_idx], dtype=torch.long).to(device)
    state = model.init_hidden(input_idx.size(0), device)
    text_generated = []
    model.eval()
    with torch.no_grad():
        for i in range(generation_length):
            predictions, state = model(input_idx, state, return_state=True)
            # predictions shape: [batch, seq, vocab]
            last_prediction = predictions[:, -1, :] 
            
            # 다음 토큰 샘플링
            next_idx = torch.multinomial(torch.softmax(last_prediction, dim=-1), num_samples=1)
            text_generated.append(idx2char[next_idx.item()])
            
            # 다음 입력을 위해 차원 조정: [1, 1]
            input_idx = next_idx
    return start_string + ''.join(text_generated)

print("\n작곡 중...")
generated_text = generate_text(model, start_string="X", generation_length=500)
print("\n🎵 생성된 악보 (ABC 표기법):")
print(generated_text)

with open("generated_song.abc", "w") as f:
    f.write(generated_text)
print("\n'generated_song.abc' 저장 완료!")
