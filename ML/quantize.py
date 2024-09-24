import numpy as np
import torch
import jax
from tqdm import tqdm

from model import LanguageModelConfig, TransformerConfig, QuantizedWeight8bit as QW8Bit
from runners import InferenceRunner, ModelRunner, sample_from_model

# Path to the checkpoint directory
CKPT_PATH = "./checkpoints"

# Configuration for the Grok-1 language model
grok_1_model = LanguageModelConfig(
    vocab_size=128 * 1024,  # Size of the vocabulary
    pad_token=0,  # Token used for padding
    eos_token=2,  # Token used for end-of-sequence
    sequence_len=8192,  # Maximum sequence length
    embedding_init_scale=1.0,  # Initial scale for embedding weights
    output_multiplier_scale=0.5773502691896257,  # Scale for output multiplier
    embedding_multiplier_scale=78.38367176906169,  # Scale for embedding multiplier
    model=TransformerConfig(
        emb_size=48 * 128,  # Embedding size
        widening_factor=8,  # Widening factor for the Transformer model
        key_size=128,  # Size of the attention key
        num_q_heads=48,  # Number of query attention heads
        num_kv_heads=8,  # Number of key-value attention heads
        num_layers=64,  # Number of Transformer layers
        attn_output_multiplier=0.08838834764831845,  # Multiplier for attention output
        shard_activations=True,  # Whether to shard activations
        # Mixture of Experts (MoE) configuration
        num_experts=8,  # Number of experts in MoE
        num_selected_experts=2,  # Number of experts to select in MoE
        # Activation sharding configuration
        data_axis="data",  # Axis for data parallelism
        model_axis="model",  # Axis for model parallelism
    ),
)

# Create a ModelRunner for training and inference
runner = ModelRunner(
    model=grok_1_model,  # Language model configuration
    bs_per_device=0.125,  # Batch size per device
    checkpoint_path=CKPT_PATH,  # Path to checkpoint directory
)

# Create dummy data for initialization
dummy_data = dict(
    inputs=np.zeros((1, 256), dtype=np.int32),  # Dummy input data
    targets=np.zeros((1, 256), dtype=np.int32),  # Dummy target data
)

runner.transform_forward = True  # Enable forward pass transformation
runner.initialize(dummy_data, (1, 1), (1, 1))  # Initialize the runner with dummy data
params = runner.load_or_init(dummy_data)  # Load or initialize model parameters

new_params = {}  # Dictionary to store updated parameters
keys = list(params.keys())  # Get the list of parameter keys

# Iterate over parameter keys and update them
for key in tqdm(keys):
    # Update the parameter key to match the desired format
    new_key = key.replace('/', '.').replace('decoder_layer_', 'decoder_layer.').replace('language_model', 'transformer')
    new_key += '.weight'  # Append '.weight' to the key

    v = list(params[key].values())[0]  # Get the parameter value

    if hasattr(v, 'scales'):
        # Determine the data type based on the scales data type
        dtype = torch.float32 if v.scales.dtype == np.float32 else torch.bfloat16
        
        # Convert weight and scales to PyTorch tensors
        weight = torch.from_numpy(np.asarray(v.weight).astype(np.float32)).to(dtype)
        scale = torch.from_numpy(np.asarray(v.scales).astype(np.float32)).to(dtype)
        
        # Handle row parallel layers with sharded scale
        if len(scale.shape) >= 2 and scale.shape[-2] != 1:
            scale = scale[..., None, :]
            weight = weight.view(*weight.shape[:-2], 8, -1, weight.shape[-1])
            weight = (weight * scale).view(*weight.shape[:-3], -1, weight.shape[-1])
        else:
            weight = weight * scale
    else:
        # Determine the data type based on the parameter data type
        dtype = torch.float32 if v.dtype == np.float32 else torch.bfloat16
        
        # Convert the parameter to a PyTorch tensor
        weight = torch.from_numpy(np.asarray(v).astype(np.float32)).to(dtype)
    
    # Transpose linear matrix if it's not an input/output embedding
    if len(weight.shape) >= 2 and 'in_out_embed' not in new_key:
        weight = weight.transpose(-1, -2).contiguous()

    if 'moe' not in new_key:
        new_params[new_key] = weight  # Store the updated parameter
    else:
        # Split MoE parameters into individual experts
        for i in range(8):
            new_key_i = new_key.replace('moe', f'moe.{i}')
            new_params[new_key_i] = weight[i]

    del params[key]  # Remove the original parameter

# Save the updated parameters to a PyTorch model file
torch.save(new_params, 'hf/pytorch_model.bin')