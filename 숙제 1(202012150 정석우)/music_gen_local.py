import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from tqdm import tqdm

# 1. 샘플 데이터 (아일랜드 민요 스타일의 간단한 ABC 악보)
# 실제 프로젝트에서는 더 많은 데이터를 파일로 읽어오면 성능이 좋아집니다.
sample_abc = """
X:1
T:The Wind that Shakes the Barley
M:4/4
L:1/8
R:reel
K:Dmaj
|:A2AB AFDF|G2GA G2FG|A2AB AFDF|E2EF E2FG:|
|:Addc d2df|e2ed e2fg|afge fdec|d2dc d2dB:|

X:2
T:The Blacksmith
M:4/4
L:1/8
R:reel
K:Dmaj
|:d2fd Adfd|ceec d2Bc|d2fd Adfd|eaag f2ef:|
|:g2gf gaba|f2fe f2af|g2gf gfed|ceec d2Bc:|
"""

# 2. 데이터 전처리
songs = [sample_abc] * 10 # 학습을 위해 데이터를 반복 (실제로는 수천 곡이 필요함)
songs_joined = "\n\n".join(songs)
vocab = sorted(set(songs_joined))
char2idx = {u: i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

def vectorize_string(string):
    return np.array([char2idx[char] for char in string])

vectorized_songs = vectorize_string(songs_joined)

def get_batch(vectorized_songs, seq_length, batch_size):
    n = vectorized_songs.shape[0] - 1
    idx = np.random.choice(n - seq_length, batch_size)
    input_batch = [vectorized_songs[i: i + seq_length] for i in idx]
    output_batch = [vectorized_songs[i + 1: i + seq_length + 1] for i in idx]
    return torch.tensor(input_batch, dtype=torch.long), torch.tensor(output_batch, dtype=torch.long)

# 3. 모델 정의
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super(LSTMModel, self).__init__()
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

# 4. 설정 및 학습
device = torch.device("cpu") # 로컬 CPU 환경
vocab_size = len(vocab)
embedding_dim = 64
hidden_size = 256
batch_size = 4
seq_length = 50
learning_rate = 0.01
num_iterations = 500

model = LSTMModel(vocab_size, embedding_dim, hidden_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

print("학습 시작...")
model.train()
for i in tqdm(range(num_iterations)):
    x_batch, y_batch = get_batch(vectorized_songs, seq_length, batch_size)
    optimizer.zero_grad()
    y_hat = model(x_batch)
    loss = criterion(y_hat.view(-1, vocab_size), y_batch.view(-1))
    loss.backward()
    optimizer.step()

# 5. 음악 생성
def generate_text(model, start_string="X:", length=500):
    input_idx = torch.tensor([[char2idx[s] for s in start_string]], dtype=torch.long)
    state = model.init_hidden(1, device)
    generated = [start_string]
    model.eval()
    with torch.no_grad():
        for _ in range(length):
            preds, state = model(input_idx, state, return_state=True)
            next_char_idx = torch.multinomial(torch.softmax(preds[:, -1, :], dim=-1), 1)
            generated.append(idx2char[next_char_idx.item()])
            input_idx = next_char_idx
    return "".join(generated)

print("\n--- 생성된 ABC 악보 ---")
result = generate_text(model)
print(result)

with open("generated_music.abc", "w") as f:
    f.write(result)
print("\n결과가 'generated_music.abc' 파일로 저장되었습니다.")
