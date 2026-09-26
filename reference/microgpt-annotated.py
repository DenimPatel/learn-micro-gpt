"""
The most atomic way to train and inference a GPT in pure, dependency-free Python.
This file is the complete algorithm.
Everything else is just efficiency.
@karpathy
"""

# Importing standard libraries needed for the script.
# 'os' is used for file system operations, explicitly checking if a file exists.
import os       # os.path.exists
# 'math' provides mathematical functions like natural logarithm (log) and exponentiation (exp).
# These are crucial for the mathematical operations in neural networks, specifically softmax and loss calculation.
import math     # math.log, math.exp
# 'random' allows us to generate random numbers.
# This is used for initializing weights, processing data potentially in random order, and sampling during inference.
import random   # random.seed, random.choices, random.gauss, random.shuffle

# Setting the seed for the random number generator ensures reproducibility.
# By using the same seed (42), we guarantee that the random numbers generated are the same every time we run the script.
# This is important for debugging and comparing results, so "random" events are actually deterministic.
random.seed(42) # Let there be order among chaos

# -----------------------------------------------------------------------------
# DATA LOADING SECTION
# -----------------------------------------------------------------------------

# We need a dataset to train our model. Here we use a list of names.
# Check if the file 'input.txt' exists in the current directory.
if not os.path.exists('input.txt'):
    # If the file doesn't exist, we need to download it.
    # urllib is a standard library for opening URLs.
    import urllib.request
    # This URL points to a raw text file containing a list of names (one per line).
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt'
    # Download the file from the URL and save it locally as 'input.txt'.
    urllib.request.urlretrieve(names_url, 'input.txt')

# Read the content of 'input.txt'.
# 1. open('input.txt'): Opens the file.
# 2. .read(): Reads the entire file content into a single string.
# 3. .strip(): Removes leading/trailing whitespace from the whole file.
# 4. .split('\n'): Splits the huge string into a list of strings, one for each line (name).
# 5. [l.strip() ... if l.strip()]: A list comprehension that iterates through each line 'l'.
#    It strips whitespace from each line and includes it only if it's not empty.
# 'docs' becomes our dataset: a list of strings, where each string is a name.
docs = [l.strip() for l in open('input.txt').read().strip().split('\n') if l.strip()] # list[str] of documents

# Shuffle the list of documents randomly.
# This prevents the model from learning any order-dependent patterns in the dataset (like alphabetical order).
# It ensures the training batches are representative of the overall distribution.
random.shuffle(docs)
print(f"num docs: {len(docs)}")

# -----------------------------------------------------------------------------
# TOKENIZATION SECTION
# -----------------------------------------------------------------------------
# A Tokenizer translates strings (text) to discrete symbols (integers) and back.
# This model works at the character level, so each character is a token.

# Create the vocabulary of characters.
# 1. ''.join(docs): Concatenates all names into one massive string.
# 2. set(...): Finds all unique characters in that string.
# 3. sorted(...): Sorts these characters alphabetically to verify a consistent order.
# 'uchars' is a list of all unique characters appearing in the names.
uchars = sorted(set(''.join(docs))) # unique characters in the dataset become token ids 0..n-1

# We need a special token to mark the Beginning Of Sequence (BOS).
# This tells the model "we are starting a new name here".
# We assign it the next available integer index, which is equal to the number of unique characters.
BOS = len(uchars) # token id for the special Beginning of Sequence (BOS) token

# The total vocabulary size is the number of unique characters plus the one special BOS token.
vocab_size = len(uchars) + 1 # total number of unique tokens, +1 is for BOS
print(f"vocab size: {vocab_size}")

# -----------------------------------------------------------------------------
# AUTOGRAD ENGINE (Micrograd)
# -----------------------------------------------------------------------------
# This class implements a scalar-value autograd engine.
# "Autograd" stands for Automatic Gradient. It allows us to build a mathematical expression graph
# and then automatically calculate the derivatives (gradients) of the output with respect to inputs.
# This is the core mechanism that allows neural networks to learn via Backpropagation.

class Value:
    # __slots__ is a Python optimization that explicitly declares data members.
    # This saves memory by preventing the creation of a dynamic __dict__ for each instance.
    # Since we will create millions of Value objects, this memory saving is critical.
    __slots__ = ('data', 'grad', '_children', '_local_grads')

    def __init__(self, data, children=(), local_grads=()):
        # The actual scalar value (e.g., a weight, an input, or an intermediate calculation result).
        self.data = data
        
        # The gradient of the loss function with respect to this value.
        # Initially 0, it accumulates gradients from the backward pass.
        # It represents how much the final Loss would change if we increased this specific value slightly.
        self.grad = 0
        
        # A tuple of other Value objects that produced this Value.
        # For example, if z = x + y, then z._children would be (x, y).
        # This builds the connectivity of the computation graph.
        self._children = children
        
        # Stores the "local derivative" of this operation with respect to its children.
        # This is the Chain Rule piece for this specific operation.
        # e.g., for z = x * y, dz/dx = y. So local_grads might store 'y' to be multiplied during backward.
        self._local_grads = local_grads

    # Operator overloading for addition (+).
    # This acts as the forward pass for addition.
    def __add__(self, other):
        # Ensure 'other' is a Value object. If it's a number (int/float), wrap it in Value.
        other = other if isinstance(other, Value) else Value(other)
        
        # Mathematical rule: if z = x + y, then dz/dx = 1 and dz/dy = 1.
        # So the local gradient for both children is 1.
        # output = self.data + other.data
        return Value(self.data + other.data, (self, other), (1, 1))

    # Operator overloading for multiplication (*).
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        
        # Mathematical rule: if z = x * y, then dz/dx = y and dz/dy = x.
        # So the local gradient for self is 'other.data' and for other is 'self.data'.
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    # Operator overloading for power (**).
    def __pow__(self, other):
        # We assume 'other' is a constant scalar (int/float), not a Value object training parameter.
        # Mathematical rule: if z = x^c, then dz/dx = c * x^(c-1).
        return Value(self.data**other, (self,), (other * self.data**(other-1),))

    # Natural logarithm.
    def log(self):
        # Mathematical rule: if z = log(x), then dz/dx = 1/x.
        return Value(math.log(self.data), (self,), (1/self.data,))

    # Exponential function (e^x).
    def exp(self):
        # Mathematical rule: if z = e^x, then dz/dx = e^x = z.
        return Value(math.exp(self.data), (self,), (math.exp(self.data),))

    # Rectified Linear Unit (ReLU) activation function.
    # It introduces non-linearity. Output is x if x > 0, else 0.
    def relu(self):
        # Mathematical rule: if z = max(0, x), dz/dx = 1 if x > 0 else 0.
        return Value(max(0, self.data), (self,), (float(self.data > 0),))

    # Negation (-x).
    def __neg__(self): return self * -1
    
    # Reverse addition (constant + Value).
    def __radd__(self, other): return self + other
    
    # Subtraction (self - other).
    def __sub__(self, other): return self + (-other)
    
    # Reverse subtraction (constant - Value).
    def __rsub__(self, other): return other + (-self)
    
    # Reverse multiplication (constant * Value).
    def __rmul__(self, other): return self * other
    
    # Division (self / other).
    def __truediv__(self, other): return self * other**-1
    
    # Reverse division (constant / Value).
    def __rtruediv__(self, other): return other * self**-1

    # The backward pass engine.
    # This propagates gradients from the output (loss) back to the inputs (weights).
    def backward(self):
        # 1. Topological Sort:
        # We need to process nodes in an order such that we only calculate a node's gradient
        # after all the nodes that consume it have been processed.
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self) # Start sorting from the final node (loss).
        
        # 2. Backpropagation:
        # Initialize the gradient of the loss with respect to itself as 1.0 (dLoss/dLoss = 1).
        self.grad = 1
        
        # Iterate through the nodes in reverse topological order (computation graph backwards).
        for v in reversed(topo):
            # For each child of the current node v:
            # Propagate v's gradient to the child using the chain rule.
            # child.grad += (local derivative of v w.r.t child) * (global gradient of v)
            # The += is crucial because a variable might be used multiple times (multivariate chain rule).
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad

# -----------------------------------------------------------------------------
# MODEL INITIALIZATION
# -----------------------------------------------------------------------------
# Hyperparameters: Configuration settings for the model size and structure.
n_embd = 16     # Embedding dimension: How many numbers represent a single token? (Feature vector size)
n_head = 4      # Number of attention heads: How many "perspectives" does the attention mechanism have?
n_layer = 1     # Number of layers: How many times do we repeat the Attention + MLP block?
block_size = 16 # Maximum sequence length: How far back in the past can the model "see"?
head_dim = n_embd // n_head # Dimension of each individual attention head. (16 // 4 = 4)

# Initialization function for weight matrices.
# Creates a list of lists of Values initialized with random numbers from a Gaussian distribution.
# nout: number of output features (rows), nin: number of input features (cols).
# std=0.08: Standard deviation, controls the spread of initial random weights.
matrix = lambda nout, nin, std=0.08: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]

# State Dictionary: Holding all the trainable parameters of the model.
state_dict = {
    # Token Embeddings (wte): [vocab_size x n_embd]. Learnable vector for each character.
    'wte': matrix(vocab_size, n_embd),
    # Position Embeddings (wpe): [block_size x n_embd]. Learnable vector for each position index (0 to 15).
    # This allows the model to know "where" a token is in the sequence.
    'wpe': matrix(block_size, n_embd),
    # Language Model Head (lm_head): [vocab_size x n_embd]. Projects final features back to vocabulary size for prediction.
    'lm_head': matrix(vocab_size, n_embd)
}

# Initialize parameters for each layer of the Transformer.
for i in range(n_layer):
    # Attention weights: Query (wq), Key (wk), Value (wv), Output (wo).
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)
    # Feed-Forward Network (MLP) weights.
    # fc1 expands dimension by 4x (standard Transformer design), fc2 projects it back.
    state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)

# Flatten all parameters from the state_dict into a single list of Value objects.
# This makes it easy to iterate over them for the optimizer update step.
params = [p for mat in state_dict.values() for row in mat for p in row]
print(f"num params: {len(params)}")

# -----------------------------------------------------------------------------
# MODEL ARCHITECTURE
# -----------------------------------------------------------------------------
# Helper function: Linear Layer (Matrix Multiplication).
# Computes y = Wx.
# x: input vector, w: weight matrix.
def linear(x, w):
    # For each row 'wo' in the weight matrix 'w':
    #   Compute the dot product: sum(wi * xi)
    #   wi is a weight scalar, xi is an input scalar.
    # Returns a list of Value objects representing the output vector.
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]

# Helper function: Softmax Activation.
# Converts a vector of raw scores (logits) into probabilities (sum to 1).
def softmax(logits):
    # For numerical stability, subtract the max value from all logits.
    # This prevents overflow when computing exp(). (e.g., e^1000 is too big, e^0 is 1).
    # Since softmax is shift-invariant, this doesn't change the output probabilities.
    max_val = max(val.data for val in logits)
    
    # Exponentiate each shifted logit: e^(x - max).
    exps = [(val - max_val).exp() for val in logits]
    
    # Calculate normalization constant (sum of exponentials).
    total = sum(exps)
    
    # Divide each exponential by the total to get probabilities.
    return [e / total for e in exps]

# Helper function: RMSNorm (Root Mean Square Normalization).
# Normalizes the input vector to have unit variance. Stabilizes training.
# Unlike LayerNorm, it doesn't subtract the mean, only divides by RMS.
def rmsnorm(x):
    # Calculate the mean of squares: sum(x^2) / N.
    ms = sum(xi * xi for xi in x) / len(x)
    
    # Calculate the scaling factor: 1 / sqrt(mean_squares + epsilon).
    # 1e-5 is a small number to prevent division by zero.
    scale = (ms + 1e-5) ** -0.5
    
    # Scale each element of the input.
    return [xi * scale for xi in x]

# The Main GPT Model Function.
# Defines the Forward Pass: transforming input tokens into next-token predictions.
# token_id: integer id of the current token.
# pos_id: integer id of the current position in the sequence.
# keys, values: KV-cache to store past keys/values for efficient generation (though recomputed here in training).
def gpt(token_id, pos_id, keys, values):
    # 1. Embedding Stage
    tok_emb = state_dict['wte'][token_id] # Look up Token Embedding from 'wte' matrix.
    pos_emb = state_dict['wpe'][pos_id]   # Look up Position Embedding from 'wpe' matrix.
    # Add them together: Input Representation = Token Info + Position Info.
    x = [t + p for t, p in zip(tok_emb, pos_emb)]
    
    # Apply Root Mean Square Normalization before processing layers (Pre-Norm architecture).
    x = rmsnorm(x)

    # 2. Transformer Layers
    for li in range(n_layer):
        # --- Multi-Head Attention Block ---
        x_residual = x # Save input for residual connection (skip connection).
        x = rmsnorm(x) # Normalize before attention.
        
        # Calculate Query (q), Key (k), and Value (v) vectors via linear projections.
        q = linear(x, state_dict[f'layer{li}.attn_wq'])
        k = linear(x, state_dict[f'layer{li}.attn_wk'])
        v = linear(x, state_dict[f'layer{li}.attn_wv'])
        
        # Append current k and v to the history (KV cache).
        # In this specific simple implementation, we assume we process one token at a time
        # recursively, but 'keys' and 'values' accumulate history from previous steps in this sequence.
        keys[li].append(k)
        values[li].append(v)
        
        x_attn = [] # To store result of multi-head attention.
        
        # Iterate over each attention head.
        for h in range(n_head):
            # Calculate the slice indices for this head.
            hs = h * head_dim
            
            # Extract the query segment for this head.
            q_h = q[hs:hs+head_dim]
            
            # Extract keys and values history for this head. 
            # k_h is a list of vectors (one for each past timestep + current).
            k_h = [ki[hs:hs+head_dim] for ki in keys[li]]
            v_h = [vi[hs:hs+head_dim] for vi in values[li]]
            
            # Scaled Dot-Product Attention:
            # 1. Dot Product: q_h * k_h_transposed.
            #    Measures similarity between current query and all past keys.
            #    Iterate 't' over all past timesteps.
            # 2. Scale: Divide by sqrt(head_dim) to keep gradients stable.
            attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5 for t in range(len(k_h))]
            
            # 3. Softmax: Convert scores to attention weights (probabilities summing to 1).
            attn_weights = softmax(attn_logits)
            
            # 4. Weighted Sum: sum(weight * value).
            #    Aggregates information from past values based on attention weights.
            #    Iterate 'j' over the dimension of the head.
            #    Iterate 't' over all past timesteps.
            head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h))) for j in range(head_dim)]
            
            # Collect output from this head.
            x_attn.extend(head_out)
        
        # Project the concatenated head outputs back to embedding dimension using 'attn_wo'.
        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
        
        # Add Residual Connection: Input + Attention Output.
        # Allows gradients to flow directly through the network, preventing vanishing gradients.
        x = [a + b for a, b in zip(x, x_residual)]
        
        # --- Feed-Forward (MLP) Block ---
        x_residual = x # Save input for next residual connection.
        x = rmsnorm(x) # Normalize.
        
        # First Linear Layer: Expand dimension (n_embd -> 4 * n_embd).
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
        
        # Activation Function: ReLU (Rectified Linear Unit), element-wise.
        # Introduces non-linearity to the network.
        x = [xi.relu() for xi in x]
        
        # Second Linear Layer: Project back to embedding dimension (4 * n_embd -> n_embd).
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
        
        # Second Residual Connection.
        x = [a + b for a, b in zip(x, x_residual)]

    # 3. Output Head
    # Project final hidden state to vocabulary size (logits for next token).
    logits = linear(x, state_dict['lm_head'])
    return logits

# -----------------------------------------------------------------------------
# OPTIMIZER (Adam)
# -----------------------------------------------------------------------------
# We manually implement Adam (Adaptive Moment Estimation).
# It maintains two state buffers for each parameter:
#   m: Exponential moving average of gradients (momentum).
#   v: Exponential moving average of squared gradients (velocity/scaling).

learning_rate = 0.01
beta1 = 0.85 # Decay rate for first moment (momentum)
beta2 = 0.99 # Decay rate for second moment (RMS prop)
eps_adam = 1e-8 # Small epsilon to prevent division by zero

m = [0.0] * len(params) # Initialize first moment buffer with zeros
v = [0.0] * len(params) # Initialize second moment buffer with zeros

# -----------------------------------------------------------------------------
# TRAINING LOOP
# -----------------------------------------------------------------------------
num_steps = 1000 # How many optimization steps to perform
for step in range(num_steps):

    # 1. Data Preparation
    # Select a document (cyclic)
    doc = docs[step % len(docs)]
    
    # Tokenize: Convert chars to ints.
    # Prepend BOS to indicate start, Append BOS to indicate end.
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    
    # Determine sequence length (limit by block_size).
    n = min(block_size, len(tokens) - 1)

    # 2. Forward Pass
    # Initialize empty KV cache for the layers.
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    losses = []
    
    # Iterate through the sequence position by position.
    for pos_id in range(n):
        # Input: current token, Target: next token.
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
        
        # Run the model! Get logits distribution for the next token.
        logits = gpt(token_id, pos_id, keys, values)
        
        # Calculate probabilities using Softmax.
        probs = softmax(logits)
        
        # 3. Loss Calculation (Cross-Entropy Loss)
        # We want to maximize the probability of the correct 'target_id'.
        # Equivalently, we minimize the negative log likelihood (NLL).
        # loss = -log(probability_assigned_to_correct_token)
        loss_t = -probs[target_id].log()
        losses.append(loss_t)
        
    # Average the loss over the sequence tokens.
    loss = (1 / n) * sum(losses)

    # 4. Backward Pass (Backpropagation)
    # This triggers the recursive chain rule calculation.
    # It populates the .grad attribute of every weight in 'params'.
    loss.backward()

    # 5. Parameter Update (Adam Step)
    # Calculate current learning rate with linear decay (annealing).
    # Learning rate goes from 0.01 down to 0 over the training steps.
    lr_t = learning_rate * (1 - step / num_steps)
    
    for i, p in enumerate(params):
        # Update first weighted moment (momentum).
        m[i] = beta1 * m[i] + (1 - beta1) * p.grad
        
        # Update second weighted moment (squared gradients).
        v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
        
        # Bias correction (important for early steps when m/v starts at 0).
        m_hat = m[i] / (1 - beta1 ** (step + 1))
        v_hat = v[i] / (1 - beta2 ** (step + 1))
        
        # Update the parameter!
        # p.data = p.data - learning_rate * momentum / (std_dev + epsilon)
        p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
        
        # Zero gradients for the next step (Pytorch default behavior too).
        p.grad = 0

    print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f}")

# -----------------------------------------------------------------------------
# INFERENCE (Generation)
# -----------------------------------------------------------------------------
# "May the model babble back to us". We generate new names.
temperature = 0.5 # Controls randomness. Lower = more conservative, Higher = more creative/crazy.
print("\n--- inference (new, hallucinated names) ---")

for sample_idx in range(20):
    # Reset KV cache for new sample.
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    
    # Start with the Beginning Of Sequence token.
    token_id = BOS
    sample = []
    
    # Generate up to block_size tokens.
    for pos_id in range(block_size):
        # Forward pass to get logits for the next token.
        logits = gpt(token_id, pos_id, keys, values)
        
        # Apply temperature.
        # Dividing logits by temp < 1 pushes probabilities apart (peaky -> conservative).
        # Dividing logits by temp > 1 flattens probabilities (more uniform -> random).
        probs = softmax([l / temperature for l in logits])
        
        # Sample the next token index from the probability distribution.
        # weighted random choice.
        token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
        
        # If we picked the BOS token (which we treat as EOS here), stop generating.
        if token_id == BOS:
            break
            
        # Add the character to our generated string.
        sample.append(uchars[token_id])
        
    print(f"sample {sample_idx+1:2d}: {''.join(sample)}")