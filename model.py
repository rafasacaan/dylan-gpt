import torch 
import torch.nn as nn
from torch.nn import functional as F


# # Hyperparamters - gpu
# batch_size = 64      # number of independent sequences to process in parallel
# block_size = 256       # context length
# max_iters = 5_000    # training iterations
# eval_interval = 500  # how often to evaluate the loss on train and val sets
# eval_iters = 200     # how many iterations to average loss over when evaluating
# learning_rate = 3e-4 # learning rate param
# n_embed = 384         # embedding size
# n_layer = 6          # number of transformer blocks
# dropout = 0.2        # dropout rate
# n_head = 6           # number of heads
# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Hyperparamters
batch_size = 16      # number of independent sequences to process in parallel
block_size = 32       # context length
max_iters = 5_000    # training iterations
eval_interval = 100  # how often to evaluate the loss on train and val sets
eval_iters = 200     # how many iterations to average loss over when evaluating
learning_rate = 1e-3 # learning rate param
n_embed = 64         # embedding size
n_layer = 4          # number of transformer blocks
dropout = 0.0        # dropout rate
n_head = 4           # number of heads
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Set seed
torch.manual_seed(1337)

# Load data
#!wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Create mapping from characters to integers and vice versa
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]  # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string


# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


# Return a single batch 
def get_batch(split):
    # generate a small batch of data of inputs x 
    # and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(
        low=0, 
        high=len(data) - block_size, 
        size=(batch_size,))

    # get the context for each index, and stack it into
    # rows.
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    """Averages out the loss over multiple batches, in order
    to get a better estimate of the loss.
    """
    out = {}
    
    # set model to evaluation phase
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    
    # set model to training phase
    model.train()
    return out


# Define Self Attention Model
class Head(nn.Module):
    """One head of self-attention.
    """
    
    def __init__(self, head_size):
        super().__init__()
        # Create independent/parallel key and query for each token
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # (B, T, C)
        q = self.query(x) # (B, T, C)
        
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-1, -2) * C**-0.5 # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        
        # perform the weighted aggregation of the values
        v = self.value(x) # (B, T, C)
        out = wei @ v # (B, T, T) @ (B, T, C) = (B, T, C)
        return out
    
    
# Deifne Multi-Head Self Attention Model
class MultiHeadAttention(nn.Module):
    """Multiple heads of self attention in parallel"""
    
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # concatenate  the heads over the channel dimension (i.e. horizontally)
        out = torch.cat([h(x) for h in self.heads], dim=-1) # (B, T, C*num_heads)
        out = self.dropout(self.proj(out))
        return out    
       

class FeedForward(nn.Module):
    """A simple linear layer followed by a non-linearity"""
    
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout), 
        )
        
    def forward(self, x):
        return self.net(x)       
       
    
class LayerNorm(nn.Module): # (used to be BatchNorm1d)
    """Layer norm is equivalent to Batch Norm but normalizes
    over the channel dimension only (i.e. columns))."""

    def __init__(self, dim, eps=1e-5, momentum=0.1):
        super().__init__()
        self.eps = eps
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)

    def __call__(self, x):
        # calculate the forward pass
        xmean = x.mean(1, keepdim=True) # batch mean
        xvar = x.var(1, keepdim=True) # batch variance
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps) # normalize to unit variance
        self.out = self.gamma * xhat + self.beta
        return self.out

    def parameters(self):
        return [self.gamma, self.beta]


class Block(nn.Module):
    """Transformer block: communication followed by computation"""
    
    def __init__(self, n_embed, n_head):
        # n_embed: embedding size
        # n_head: number of heads
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


# Define Bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed) # final layer norm
        self.lm_head = nn.Linear(n_embed, vocab_size) # language model head


    def forward(self, idx, targets=None):
        # idx and targets are both (B, T) tensor of integers
        # remember we have 8 examples for each row in X
        # reminder: (B, T, C) = (batch, time, channel)
        
        B, T = idx.shape
        token_embed = self.token_embedding_table(idx) # -> (B, T, C)
        pos_embed = self.position_embedding_table(torch.arange(T, device=device)) # -> (T, C)
        x = token_embed + pos_embed # -> (B, T, C)
        x = self.blocks(x) # (B, T, C)
        x = self.ln_f(x)
        logits = self.lm_head(x) # -> (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            # F.cross_entropy requires a (B, C, T) shape
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets) 
            
        return logits, loss


    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop the idx to the last block_size tokens
            idx_cond = idx[:, -block_size:] # (B, T)
            
            # get predictions
            logits, loss = self(idx_cond)
    
            # focus only on the last time step 
            # i.e. last token in sequence
            logits = logits[:, -1, :] # becomes (B, C)

            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)

            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)

            # append sampled index to running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
    

# Create model    
model = BigramLanguageModel()
m = model.to(device)

# Create a pytorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
    # sample a batch of data
    xb, yb = get_batch('train')
    
    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
# Generate from model
context = torch.zeros((1,1),dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))