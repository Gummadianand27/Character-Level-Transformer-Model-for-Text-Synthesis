import torch
import torch.nn as nn
from torch.nn import functional as F

# Hyperparameters
batch_size = 64          # Number of samples per batch
block_size = 256         # Length of context for each prediction
max_iters = 5000         # Maximum training iterations
eval_interval = 500      # Evaluation interval
learning_rate = 3e-4     # Learning rate for the optimizer
device = 'cuda'          # Device to run the model ('cuda' or 'cpu')
eval_iters = 200         # Number of iterations for evaluation
n_embd = 384             # Embedding size
n_head = 6               # Number of attention heads
n_layer = 6              # Number of Transformer blocks
dropout = 0.2            # Dropout rate

# Set random seed for reproducibility
torch.manual_seed(1337)

# Load the Shakespeare dataset
with open('shakespeare.txt', 'r', encoding='utf-8') as f: 
    text = f.read()

# Load the Bible dataset (optional)
'''
with open('bible.txt', 'r', encoding='utf-8') as f: 
    text = f.read()
'''
#-------------
# Create vocabulary mappings
chars = sorted(list(set(text)))  # Unique characters in the text
vocab_size = len(chars)          # Number of unique characters

stoi = {ch: i for i, ch in enumerate(chars)}  # Character to index
itos = {i: ch for i, ch in enumerate(chars)}  # Index to character
encode = lambda s: [stoi[c] for c in s]       # Encoding function
decode = lambda l: ''.join([itos[i] for i in l])  # Decoding function

# Split data into training and validation sets
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))        # 90% training, 10% validation
train_data = data[:n]
val_data = data[n:]

# Function to load batches of data for training and validation
def get_batch(split):
    # Use training data if split is 'train', else use validation data
    data = train_data if split == 'train' else val_data
    # Randomly select starting indices for each sample in the batch
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # Generate input (x) and target (y) sequences from these starting indices
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    # Move data to the appropriate device (CPU or GPU)
    x, y = x.to(device), y.to(device)
    return x, y

# Function to estimate loss during training and validation
@torch.no_grad()  # Disables gradient calculations to speed up evaluation
def estimate_loss():
    out = {}
    model.eval()  # Switch model to evaluation mode (disables dropout)
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)  # Compute model outputs and loss
            losses[k] = loss.item()
        out[split] = losses.mean()  # Average loss over iterations for stability
    model.train()  # Return to training mode
    return out

# Define an individual head for self-attention
class Head(nn.Module):
    ''' One head of attention'''
    def __init__(self, head_size):
        super().__init__()
        # Linear layers for computing query, key, and value representations
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # Lower-triangular mask to prevent attending to future tokens
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        # Calculate key, query, and value matrices
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute scaled dot-product attention with masked future tokens scores ("affinities") 
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # Apply attention weights to values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

# Multi-head attention with multiple heads in parallel
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        # Create multiple heads
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        # Project the concatenated outputs of all heads
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Concatenate outputs from each head and apply projection
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

# Feed-forward network with two linear layers and ReLU activation
class FeedFoward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        # Two-layer neural network with ReLU and dropout
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

# Transformer block with attention and feed-forward layers
class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        # Set head size for multi-head attention
        head_size = n_embd // n_head
        # Self-attention and feed-forward layers
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        # Layer normalization for stable training
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # Apply residual connections, layer norm, and attention/FF layers
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
        
# Main GPT-style language model
class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # Embedding layers for tokens and positions
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # Stack of Transformer blocks
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
         # Layer normalization and linear projection for final output
        self.ln_f = nn.LayerNorm(n_embd) 
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # Initialize weights for stability
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights of a module using Kaiming normal initialization.

        This is used to initialize the weights of the model. It is similar to the
        initialization used in the original GPT paper, but with the addition of
        bias initialization.
        """
        if isinstance(module, nn.Linear):
            # Kaiming normal initialization for linear layers
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                # Initialize bias as zero
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # Kaiming normal initialization for embedding layers
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # Token and position embeddings combined
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        # Apply Transformer blocks
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        # Project to vocabulary size for logits
        logits = self.lm_head(x) # (B,T,vocab_size)

        # Compute cross-entropy loss if targets are provided
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # Limit context / Crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions i.e. Compute logits for current context
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # Sample next token from probability distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

# Instantiate and move model to device
model = GPTLanguageModel()
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Training loop
for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))

# Uncomment to save outputs
# Write the generated Shakespeare-style text to 'fakespeare.txt'
# open('fakespeare.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))

# Write the generated Bible-style text to 'bibble.txt'
# open('bibble.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))
