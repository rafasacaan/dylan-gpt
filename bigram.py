import torch 
import torch.nn as nn
from torch.nn import functional as F


# Hyperparamters
batch_size = 32      # number of independent sequences to process in parallel
block_size = 8       # context length
max_iters = 3_000    # training iterations
eval_interval = 300  # how often to evaluate the loss on train and val sets
eval_iters = 200     # how many iterations to average loss over when evaluating
learning_rate = 1e-2 # learning rate param
n_embed = 32         # embedding size
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


# Define Bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        # each token directly reads off the logits for the next token 
        # from a look-up table.
        # The output will diredtly be the embedding, that is why it has
        # an output size of vocab_size
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both (B, T) tensor of integers
        # remember we have 8 examples for each row in X
        
        # reminder: (B, T, C) = (batch, time, channel)
        logits = self.token_embedding_table(idx) # -> (B, T, C)

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
            # get predictions
            logits, loss = self(idx)
    
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