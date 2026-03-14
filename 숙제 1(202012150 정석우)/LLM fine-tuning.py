import os
import json
import numpy as np
from tqdm import tqdm
import torch
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from lion_pytorch import Lion

# API 키 및 설정 (원본 코드 구조 유지)
OPENROUTER_API_KEY = "Your_OpenRouter_Key" # 나중에 본인 키를 넣어주세요
OPIK_API_KEY = "Your_Opik_Key"

# 1. 모델 및 토크나이저 준비 (현재 환경을 고려하여 135M 모델로 실행)
# 원본: LiquidAI/LFM2-1.2B (CPU에서는 수 시간 소요 및 메모리 부족 가능성)
print("모델 로딩 중... (실행을 위해 135M 모델을 사용합니다)")
model_id = "HuggingFaceTB/SmolLM-135M" # 로컬 테스트용
tokenizer = AutoTokenizer.from_pretrained(model_id)
# pad_token 설정 (SmolLM은 pad_token이 없을 수 있음)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_id)

# 채팅 템플릿 설정 (원본과 동일)
template_without_answer = "<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
template_with_answer = template_without_answer + "{answer}<|im_end|>\n"

def chat(question, max_new_tokens=32, temperature=0.7, only_answer=False):
    prompt = template_without_answer.format(question=question)
    input_ids = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(**input_ids, do_sample=True, max_new_tokens=max_new_tokens, 
                                 temperature=temperature, pad_token_id=tokenizer.eos_token_id)
    
    output_tokens = outputs[0]
    if only_answer:
        output_tokens = output_tokens[input_ids['input_ids'].shape[1]:]
    
    return tokenizer.decode(output_tokens, skip_special_tokens=True)

# 2. LoRA 세팅 (원본과 동일)
def apply_lora(model):
    lora_config = LoraConfig(
        r=8,
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"], # SmolLM 구조에 맞게 조정 (일반적인 LoRA 대상)
    )
    return get_peft_model(model, lora_config)

def forward_and_compute_loss(model, tokens, mask, context_length=512):
    tokens = tokens[:, :context_length]
    mask = mask[:, :context_length]
    
    x = tokens[:, :-1]
    y = tokens[:, 1:]
    mask = mask[:, 1:]
    
    logits = model(x).logits
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="none")
    # 마스크가 True인 부분만 Loss 계산
    return loss[mask.reshape(-1)].mean()

# 3. 데이터 로더 모킹 (mitdeeplearning 대용)
class MockDataloader:
    def __init__(self, style="yoda"):
        self.data = [
            {"instruction": "What is the dark side?", "response_style": "Hard to see, the dark side is. Fear, anger, aggression; the dark side of the Force are they."},
            {"instruction": "How do I become a Jedi?", "response_style": "Patience you must have, my young Padawan. Strong with the Force, you will become."},
            {"instruction": "Is Yoda strong?", "response_style": "Size matters not. Look at me. Judge me by my size, do you?"},
        ] * 10 # 30개 데이터 생성
        self.idx = 0
        
    def __iter__(self):
        return self
        
    def __next__(self):
        if self.idx >= len(self.data):
            self.idx = 0
            raise StopIteration
        batch = {k: [v] for k, v in self.data[self.idx].items()}
        self.idx += 1
        return batch

# 4. 학습 루프 (원본과 동일)
def train(model, dataloader, tokenizer, max_steps=5, context_length=128, learning_rate=1e-4):
    model = apply_lora(model)
    optimizer = Lion(model.parameters(), lr=learning_rate)
    
    print("\n파인튜닝 시작!")
    for step, batch in enumerate(dataloader):
        question = batch["instruction"][0]
        answer = batch["response_style"][0]
        
        text = template_with_answer.format(question=question, answer=answer)
        # return_offsets_mapping은 모델마다 다를 수 있으므로 간단하게 구현
        ids = tokenizer(text, return_tensors="pt")
        input_ids = ids["input_ids"]
        
        # 간단한 마스크 생성: 대략적으로 assistant 답변 부분만 마스킹
        prompt_len = len(tokenizer(template_without_answer.format(question=question))["input_ids"])
        mask = torch.zeros_like(input_ids, dtype=torch.bool)
        mask[:, prompt_len:] = True
        
        loss = forward_and_compute_loss(model, input_ids, mask, context_length)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 1 == 0:
            print(f"Step {step} | Loss: {loss.item():.4f}")
            
        if step >= max_steps:
            break
            
    return model

# 메인 실행
train_loader = MockDataloader(style="yoda")
model = train(model, train_loader, tokenizer, max_steps=5) # 시간을 위해 5스텝만 진행

print("\n--- 파인튜닝 후 답변 테스트 ---")
print("Q: What is a good story about tennis?")
print("A:", chat("What is a good story about tennis?", only_answer=True, max_new_tokens=30))

# 5. 최종 결과 계산 (원본과 동일한 수식)
print("\n--- 대회 제출용 최종 결과 계산 중 ---")
yoda_test_text = "The dark side of the force, hard to see it is. Patient, you must be."
tokens = tokenizer(yoda_test_text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**tokens)
    logits = outputs.logits[:, :-1]
    targets = tokens.input_ids[:, 1:]
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

print(f"==========================================")
print(f"🎉 Yoda test loglikelihood: {loss.item():.2f}")
print(f"==========================================")
