import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import util
import magic
from encoder import encode, decode
from collections import OrderedDict

# We will compute a single token to go after this prompt:
prompt = "The quick brown fox jumped over the lazy"

# The PyTorch GPT-2 model parameters, downloaded from https://huggingface.co/gpt2/blob/main/config.json
config = json.load(open('./config.json'))

# We convert chunks of text into tokens. This is the number of how many unique
# chunks of text that are recognizable as tokens.
#
# The number '50257' comes from 1 + 256 + 50,000:
#
# - 1: START/STOP token
# - 256: 256 arbitrary "byte" tokens to represent any unknown/unicode characters
# - 50,000: automatically computed frequently-occurring blobs of text.
#   Often, these may even be whole words.
vocab_size: int = config["vocab_size"]
# The dimensionality of the space that GPT-2 has to recognize relationships between words.
#
# Visualize a map. You want to put similar meaning words close together, so maybe
# all of the "water" words (e.g. river, riverbank, stream) in one area.
#
# Instead of using a 2D point with 2 dimensions to represent a word's location in
# this space, we use a point with `n_embd` dimensions to represent a word's location
# in this space.
num_embeddings: int = config["n_embd"]
# The number of tokens the model can handle for the total input.
#
# The transformers (as defined in "Attention is All You Need") can theoretically
# support infinite context, but it is limited here for practical reasons.
#
# When training the model, we take this many tokens out from chunks of text
# to train it about learning how to predict future word sequences.
num_positions: int = config["n_positions"]
# The maximum number of tokens that the self-attention mechanism will attend to.
num_self_attenuation_context_tokens: int = config["n_ctx"]
# Number of `Block`s chained to eachother.
num_layers: int = config["n_layer"]
# Configures the normalization function to be smoother or sharper.
layer_norm_epsilon: float = config["layer_norm_epsilon"]
# Number of heads for each attention head.
num_heads: int = config["n_head"]

# Before we begin, let's get deterministic.
torch.manual_seed(2)

# The PyTorch GPT-2 model downloaded from https://huggingface.co/gpt2/blob/main/pytorch_model.bin
model: OrderedDict = torch.load('./pytorch_model.bin')
#
# The model is composed of a set of (key, value) pairs. They look like this:
#
#     (wte.weight, torch.Tensor([...]))
#     (wpe.weight, torch.Tensor([...]))
#     (h.0.ln_1.weight, torch.Tensor([...]))
#     (h.0.ln_1.bias, torch.Tensor([...]))
#     (h.0.attn.bias, torch.Tensor([...]))
#     ...
#
# As you can see, their keys form a sort of directory hierarchy whilst being
# a simple key-value store (similar to S3 objects).

# 'wte' stands for 'word_to_embedding', albeit these are technically tokens.
#
# Recall that an embedding is basically the model's understanding of a word,
# in an `n_embd`-dimensional space.
word_to_embedding = nn.Embedding(vocab_size, num_embeddings)
util.load_from_state_dict(word_to_embedding, model, "wte")

# Convert the textual prompt to tokens
prompt_tokens = torch.tensor(encode(prompt))

# Convert the tokens into embeddings.
#
# Take note of `.forward(x)`: this is unidiomatic.
#
# It's more proper to see `module(x)` instead of `module.forward(x)`.
# I find the latter more understandable as it's basically the former.
input_embeddings = word_to_embedding.forward(prompt_tokens)

# With every word, we include position information with it. For example,
#
# in:  "The quick brown fox   jumps over the lazy  dog "
# emb:  464 2068  7586  21831 11687 625  262 16931 3290
# pos:   0    1     2    3      4    5    6   7     8
#
# We use `word_to_embedding` and `word_position_encoding` to convert these
# numbers into meaningful numbers for the neural network. Then, we add them
# so that we combine the embeddings with the positional information.

# Load the word position encodings
word_position_encoding = nn.Embedding(num_positions, num_embeddings)
util.load_from_state_dict(word_position_encoding, model, "wpe")

# Turn positional information into vectors
position_ids = torch.arange(start=0, end=len(prompt_tokens), dtype=torch.long)
position_embeddings = word_position_encoding.forward(position_ids)

# Combine the input embedding with the position embedding to meaningfully merge the two
positionally_aware_embeddings = input_embeddings + position_embeddings

# Now, we feed these encodings to each block.
input = positionally_aware_embeddings  # Re-used input passed to each block

# Before: `input.size() = torch.Size([8, 768])`
# After:  `input.size() = torch.Size([1, 8, 768])`
#
# The very first value is simply the batch number. We only have one batch: our initial prompt.
# I've opted to keep the batch in-place rather than rip out all the places it was used for
# parity and easier reference-ibility with other parent code.
input = input.unsqueeze(0)

for block_idx in range(num_layers):

    # Normalize the values prior to feeding them to our self-attention mechanism
    attention_normalization = nn.LayerNorm(num_embeddings, eps=layer_norm_epsilon)
    util.load_from_state_dict(attention_normalization, model, f"h.{block_idx}.ln_1")
    normalized = attention_normalization.forward(input)

    # Load and utilize the multi-headed attention mechanism
    attention = magic.Attention(num_embeddings, num_self_attenuation_context_tokens, num_heads, scale=True)
    util.load_from_state_dict(attention, model, f"h.{block_idx}.attn")
    attention_results, _ = attention.forward(normalized)

    # In training, this sort of addition is visible to the optimizer.
    #
    # It basically allows back-propagation to be faster and work more in parallel across
    # the numerous blocks.
    #
    # Since this is done during training, we must also do it here.
    input = input + attention_results

    # Normalize values for the multiplayer perceptron
    mlp_normalization = nn.LayerNorm(num_embeddings, eps=layer_norm_epsilon)
    util.load_from_state_dict(mlp_normalization, model, f"h.{block_idx}.ln_2")
    normalized = mlp_normalization.forward(input)

    # Load the multilayer perceptron. This performs a large amount of the thinking.
    num_thinkers = 4 * num_embeddings  # I guess this was arbitrarily chosen
    mlp = magic.MLP(num_thinkers, num_embeddings)
    util.load_from_state_dict(mlp, model, f"h.{block_idx}.mlp")
    mlp_thoughts = mlp.forward(normalized)

    # The aforementioned back-propagation optimization trick
    input = input + mlp_thoughts

# Normalize the final thoughts
final_normalization = nn.LayerNorm(num_embeddings, eps=layer_norm_epsilon)
util.load_from_state_dict(final_normalization, model, f"ln_f")
final_thoughts = final_normalization.forward(input)

# Perform some final thinking
wte_shape = word_to_embedding.weight.shape
assert wte_shape[0] == vocab_size
assert wte_shape[1] == num_embeddings
decoder = nn.Linear(num_embeddings, vocab_size, bias=False)
decoder.weight = word_to_embedding.weight

decoded_logits = decoder.forward(final_thoughts)

# Undo that `.unsqueeze(0)` from earlier
decoded_logits = decoded_logits[0]

# Now `decoded_logits` is a tensor of `.size() = [8, 50257]`.
#
# This is a multidimensional array: it's an array of 8 elements, where each element
# is an array of 50257 values. These "arrays of 50257 values" are the likelihoods that
# a given word is next.
#
# Each element in this array is predicting the likelihood of the next word:
#
#     [
#       [given "The", we think "quick" is 0.69% likely],
#       [given "The quick", we think "brown" is 4.20% likely],
#       [given "The quick brown", we think "fox" is 13.37% likely],
#       ...
#     ]
#
# So now, we grab the last element from this array to predict the final word:
next_word_predictions = decoded_logits[-1]

# Softmax the probabilities to essentially normalize them
logits_probability = F.softmax(next_word_predictions, dim=-1)

# Pick a random next word based on the distribution.
next_token = torch.multinomial(logits_probability, num_samples=1).tolist()

next_word = decode(next_token)
print(prompt + next_word)
