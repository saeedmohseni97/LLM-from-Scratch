
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt


# loading The Dataset
with open('/home/saeedmohseni/Desktop/LLM/Dataset/text.txt', 'r', encoding='utf-8') as f:
#with open('input.txt', 'r', encoding='utf-8') as f:
  text = f.read()


print("length of dataset in characters: ", len(text))


# hyperparameters:
learning_rate = 3e-4
batch_size = 64
n_layer = 4
n_head = 4
context_size = 128;
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200;
n_embed =256
dropout = 0.2




# Properties of the given text, unique elements and the vocabulary size.
chars = sorted(list(set(text)))
vocabulary_size = len(chars)
print(f"unique characters are: {''.join(chars)}")
print(f"\n number of unique characters in the text: {vocabulary_size}")


# Tokenizing the characters:
encoder = lambda text: [chars.index(c) for c in text]
decoder = lambda tokenized_text: ''.join([chars[token] for token in tokenized_text])

print(encoder("This is Saeed"))
print(decoder(encoder("This is Saeed")))


data = torch.tensor(encoder(text), dtype=torch.long)
print(data.shape, data.dtype)
print(data[0:99])


train_data = data[:int(0.9*len(data))]
val_data = data[int(0.9*len(data)):]
print(train_data.shape, val_data.shape)
print(val_data.shape, data.dtype)


@torch.no_grad()
def get_loss():
  loss ={'train': 0, 'val':0}
  model.eval()
  for split in ['train', 'val']:
    for i in range(eval_iters):
      x, y = get_batch(split, batch_size)
      logits, temp = model(x, y)
      loss[split] = loss[split] + temp.item()
    loss[split]  = loss[split] / eval_iters
  model.train()
  return loss



def get_batch(indicator, batch_size):
  data = train_data if indicator == 'train' else val_data
  ix = torch.randint(len(data) - context_size, (batch_size,))
  x = torch.stack([data[j:j+context_size] for j in ix])
  y = torch.stack([data[j+1:j+context_size+1] for j in ix])
  x, y = x.to(device), y.to(device)
  return x, y



# Head module (self attention):
class Head(nn.Module):
  def __init__(self, head_size):
    super().__init__()
    self.key = nn.Linear(n_embed, head_size, bias=False)
    self.query = nn.Linear(n_embed, head_size, bias=False)
    self.value = nn.Linear(n_embed, head_size, bias=False)
    self.dropout = nn.Dropout(dropout)
    self.register_buffer('tril', torch.tril(torch.ones(context_size, context_size)))

  def forward(self, x):
    B, T, C = x.shape
    k = self.key(x)
    q = self.query(x)
    v = self.value(x)
    wei = q @ k.transpose(-2, -1) * C**-0.5
    wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
    wei = F.softmax(wei, dim=-1)
    wei = self.dropout(wei)
    out = wei @ v
    return out

class MultiHeadAttention(nn.Module):
  def __init__(self, num_heads, head_size):
    super().__init__()
    self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
    self.proj = nn.Linear(head_size * num_heads, n_embed)
    self.dropout = nn.Dropout(dropout)
  def forward(self, x):
    return self.dropout(self.proj(torch.cat([h(x) for h in self.heads], dim=-1)))


class FeedForward(nn.Module):
  def __init__(self, n_embed):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(n_embed, 4 * n_embed),
        nn.ReLU(),
        nn.Linear(4 * n_embed, n_embed),
        nn.Dropout(dropout)
    )

  def forward(self, x):
    return self.net(x)

class Block(nn.Module):
  def __init__(self, n_embed, n_head):
    super().__init__()
    head_size = n_embed // n_head
    self.sa = MultiHeadAttention(n_head, head_size)
    self.ffwd = FeedForward(n_embed)
    self.ln1 = nn.LayerNorm(n_embed)
    self.ln2 = nn.LayerNorm(n_embed)

  def forward(self, x):
    x = x + self.sa(self.ln1(x))
    x = x + self.ffwd(self.ln2(x))
    return x




class BigramLM(nn.Module):
  def __init__(self):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocabulary_size, n_embed)
    self.position_embedding_table = nn.Embedding(context_size, n_embed)
    self.lm_head = nn.Linear(n_embed, vocabulary_size)
    self.blocks = nn.Sequential( *[Block(n_embed, n_head) for _ in range(n_layer)] )
    self.ln_f = nn.LayerNorm(n_embed)
    #self.blocks = nn.Sequential(
    #    Block(n_embed, n_head),
    #    Block(n_embed, n_head),
    #    Block(n_embed, n_head),

    self.sa_head = MultiHeadAttention(4, n_embed//4)
    self.ffwd = FeedForward(n_embed)

  def forward(self, idx, targets=None):
    B, T = idx.shape
    token_embed = self.token_embedding_table(idx)
    pos_embed = self.position_embedding_table(torch.arange(T, device=device))
    token_embed = token_embed + pos_embed
    x = self.blocks(token_embed)
    x = self.ln_f(x)
    logits = self.lm_head(x) # (B,T,C)
    B, T, C = logits.shape

    #logits = logits @ torch.triu(torch.mean(torch.ones(B,T), dim=1))

    if targets is None:
      loss = None
    else:
      logits = logits.view(B*T, C)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits, targets)
    return logits, loss

  def generate(self, idx, max_new_tokens):
    for _ in range(max_new_tokens):
      truncate_idx = idx[:, -context_size:]
      logits, loss = self(truncate_idx)
      logits = logits[:, -1, :]
      probs = F.softmax(logits, dim=-1)
      next_char = torch.multinomial(probs, num_samples=1)
      idx = torch.cat((idx, next_char), dim=1)
    return idx

x, y = get_batch('train', batch_size)
model = BigramLM().to(device)
logits, loss = model(x, y)
print(logits.shape)
print(loss)
print((model.generate(torch.zeros((1, 1), dtype=torch.long, device=device), max_new_tokens=100)))
print(decoder(model.generate(torch.zeros((1, 1), dtype=torch.long, device=device), max_new_tokens=100)[0].tolist()))
#





# optimization:
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)


epoch = 5000

train_loss = np.zeros([epoch,1])
val_loss = np.zeros([epoch,1])

for steps in range(epoch):
  x, y = get_batch('train', batch_size)
  logits, loss = model(x, y)
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()
  
  loss = get_loss()
  train_loss[steps,0] = loss['train']
  val_loss[steps,0] = loss['val']
  print(steps)
  if (steps+1) % eval_iters == 0:
    loss = get_loss()
    print(f"step: {steps+1}, train loss: {loss['train']:.4f}, val loss: {loss['val']:.4f}")


train_loss = train_loss.flatten()
val_loss = val_loss.flatten()
epochs = np.arange(1, len(train_loss) + 1)

# --- Plot ---
plt.style.use("seaborn-v0_8-whitegrid")  # clean style

plt.figure(figsize=(7, 4.5))
plt.plot(epochs, train_loss, label="Train Loss", linewidth=2)
plt.plot(epochs, val_loss, label="Validation Loss", linewidth=2, linestyle="--")

plt.title("Training and Validation Loss over Epochs", fontsize=14, fontweight='bold')
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.legend(frameon=True, fontsize=10)
plt.tight_layout()


print((model.generate(torch.zeros((1, 1), dtype=torch.long, device=device), max_new_tokens=100)))
print(decoder(model.generate(torch.zeros((1, 1), dtype=torch.long, device=device), max_new_tokens=200)[0].tolist()))


torch.save(model.state_dict(), '../SavedModels/model.pth')



model.load_state_dict(torch.load('../SavedModels/model.pth', map_location=torch.device('cpu')))


print((model.generate(torch.zeros((1, 1), dtype=torch.long, device=device), max_new_tokens=100)))
print(decoder(model.generate(torch.zeros((1, 1), dtype=torch.long, device=device), max_new_tokens=500)[0].tolist()))





