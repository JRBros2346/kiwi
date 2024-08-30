import torch
import torch.nn as nn
from torch.nn import functional as F
import pickle
import csv
from transformers import AutoTokenizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
block_size = 128
batches = 32
train_loop = 5000
learning_rate = 5e-4
train_split = 0.8
debug_loop = 500
embeds = 512
heads = 4
layers = 4
dropout = 0.2

tokenizer = AutoTokenizer.from_pretrained('gpt2')
vocab_size = len(tokenizer)

encode = lambda s: tokenizer.encode(s, add_special_tokens=True)
decode = lambda l: tokenizer.decode(l, skip_special_tokens=True)

# chars = []
# text = ""
# with open('datasets/dialogs.csv', 'r', encoding='utf-8') as f:
#     reader = csv.reader(f, delimiter='\t')
#     for row in reader:
#         text += f"developer: {row[0]}\nkivi: {row[1]}\n"
#     chars = sorted(set(text))
# print(text[:100])
# print(chars)
# vocab_size = len(chars)
# print(vocab_size)

# str2int = { ch:i for i,ch in enumerate(chars) }
# int2str = { i:ch for i,ch in enumerate(chars) }
# encode = lambda s: [ str2int[c] for c in s.lower() ]
# decode = lambda l: ''.join([ int2str[i] for i in l ])
# data = torch.tensor(encode(text), dtype=torch.long).to(device)
# print(data[:100])

def load_half_dataset(file):
    with open(file, 'r', encoding='utf-8') as f:
        f.seek(0, 2)
        half = f.tell() // 200
        f.seek(0)
        data = f.read(half)
    return data

train_data = load_half_dataset("datasets/shakespearean.txt")
val_data = load_half_dataset("datasets/shakespearean.txt")
encoded_train = torch.tensor(encode(train_data), dtype=torch.long)
encoded_val = torch.tensor(encode(val_data), dtype=torch.long)

# n = int(train_split*len(data))
# train = data[:n]
# valid = data[n:]

def get_batch(split):
    data = encoded_train if split=='train' else encoded_val
    if data.size(0) > block_size:
        ix = torch.randint(0, data.size(0) - block_size, (batches,))
        x = torch.stack([data[i:i+block_size] for i in ix])
        y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    else:
        raise ValueError("Dataset size is too small for the requested block and batch sizes.")
    x, y = x.to(device), y.to(device)
    return x, y

x, y = get_batch('train')
print('inputs:')
print(x)
print('target:')
print(y)

class MHFlashAttn(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(embeds//heads, head_size, bias=False, dtype=torch.float16)
        self.query = nn.Linear(embeds//heads, head_size, bias=False, dtype=torch.float16)
        self.value = nn.Linear(embeds//heads, head_size, bias=False, dtype=torch.float16)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        B, T, C = x.shape
        x = x.view(B, T, heads, C//heads)
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-.5
        wei = wei.masked_fill(self.tril[:T, :T]==0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.mha_flash_attn = MHFlashAttn(head_size)
        self.proj = nn.Linear(head_size, embeds)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        out = self.mha_flash_attn(x)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embeds, 4*embeds),
            nn.GELU(),
            nn.Linear(4*embeds, embeds),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, embeds, heads):
        super().__init__()
        head_size = embeds
        # x_size = torch.tensor([batches, block_size, heads, embeds])
        self.sa = MultiHeadAttention(heads, head_size)
        self.ffwd = FeedForward(embeds)
        self.ln1 = nn.LayerNorm(embeds, head_size, dtype=torch.float16)
        self.ln2 = nn.LayerNorm(embeds, head_size, dtype=torch.float16)
    def forward(self, x):
        y = self.sa(x)
        print(y.shape)
        x = self.ln1(x + y)
        y = self.ffwd(x)
        x = self.ln2(x + y)
        return x

class Kiwi(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embeds, dtype=torch.float16)
        self.position_embedding = nn.Embedding(block_size, embeds, dtype=torch.float16)
        self.blocks = nn.Sequential(*[Block(embeds, heads=heads) for _ in range(layers)])
        self.ln_final = nn.LayerNorm(embeds)
        self.head = nn.Linear(embeds, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, index, targets=None):
        B, T = index.shape

        tok = self.token_embedding(index)
        pos = self.position_embedding(torch.arange(T, device=device))
        x = tok + pos
        x = self.blocks(x)
        x = self.ln_final(x)
        logits = self.head(x)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
        
    def generate(self, index, max_new_tokens):
        for iter in range(max_new_tokens):
            index_crop = index[:, -block_size:]
            # print(index_crop)
            logits, loss = self.forward(index_crop)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            index_next = torch.multinomial(probs, num_samples=1)
            index = torch.cat((index, index_next), dim=1)
            print(f"\r<{'*' * int(10 * iter/max_new_tokens)}{' ' * (10 - int(10*iter/max_new_tokens))}>", end='', flush=False)
        print("\r<**********>")
        return index

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(debug_loop)
        for k in range(debug_loop):
            if encoded_val.size(0) > block_size:
                ix = torch.randint(0, encoded_val.size(0) - block_size, (batches,))
                x = torch.stack([encoded_val[i:i+block_size] for i in ix])
                x = torch.stack([encoded_val[i+1:i+block_size+1] for i in ix])
            else:
                raise ValueError("Dataset size is too small for the requested block and batch sizes.")
            logits, loss = model(x.to(device), y.to(device))
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def train_model(model, train_loop, debug_loop):
    optim = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    for iter in range(train_loop):
        if iter % debug_loop == 0:
            losses = estimate_loss(model)
            print(f'step: {iter}, train loss: {losses['train']:.6f}, valid loss: {losses['val']:.6f}')
        if encoded_train.size(0) > block_size:
            ix = torch.randint(0, encoded_train.size(0) - block_size, (batches,))
            x = torch.stack([encoded_train[i:i+block_size] for i in ix])
            y = torch.stack([encoded_train[i+1:i+block_size+1] for i in ix])
        else:
            raise ValueError("Dataset size is too small for the requested block and batch sizes.")
        logits, loss = model.forward(x.to(device), y.to(device))
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
    print(loss.item())

def generate(model, max_tokens, prompt=None):
    ctx = torch.zeros((1, 1), dtype=torch.long, device=device) if prompt is None else torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    # print(ctx)
    generated = decode(model.generate(ctx, max_new_tokens=max_tokens)[0].tolist())
    print(generated)
    return generated

def export_model(model, file):
    with open(file, 'wb') as f:
        pickle.dump(model, f)

def import_model(file):
    return pickle.load(open(file, 'rb'))

model = Kiwi(vocab_size).to(device)
# model = import_model('models/dialogs.pkl')
train_model(model, train_loop, debug_loop)
export_model(model, 'models/shakespear.pkl')
open('generated/shakespear-kivi.txt', 'a').write(generate(model, 100, prompt='Developer: What is your name?\n'))