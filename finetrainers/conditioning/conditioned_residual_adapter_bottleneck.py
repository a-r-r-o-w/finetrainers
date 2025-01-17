import torch.nn as nn
import torch

# This class takes the conditioned multichannel video tensor in patchified 
# Downsamples it using a bottle neck layer
# To adapt the input into the diffusion transformer.
# You want to use the patchified tensor non conditioned tensor to pass in as a residual before passing this off to the transformer.
class ConditionedResidualAdapterBottleneck(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,  # New parameter for output dimension
        bottleneck_dim: int = None,
        adapter_dropout: float = 0.1,
        adapter_init_scale: float = 1e-3,
    ):
        """
        Args:
            input_dim: Size of input dimension
            output_dim: Size of output dimension
            adapter_dropout: Dropout probability
            adapter_init_scale: Initial scale for adapter layer parameters
            use_residual: Whether to use residual connection (only if input_dim == output_dim)
        """
        super().__init__()
            
        # Down projection
        self.down_proj = nn.Linear(input_dim, bottleneck_dim)
        
        # Activation and dropout
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(adapter_dropout)
        
        # Up projection (now projects to output_dim instead of input_dim)
        self.up_proj = nn.Linear(bottleneck_dim, output_dim)
        
        # Initialize weights
        self.down_proj.weight.data.normal_(mean=0.0, std=adapter_init_scale)
        self.down_proj.bias.data.zero_()
        self.up_proj.weight.data.normal_(mean=0.0, std=adapter_init_scale)
        self.up_proj.bias.data.zero_()

    def forward(self,residual_x:torch.tensor, conditioned_x: torch.Tensor) -> torch.Tensor:    
        
        # Down projection
        hidden_states = self.down_proj(conditioned_x)
        
        # Activation and dropout
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        # Up projection to new dimension
        hidden_states = self.up_proj(hidden_states)
        

        hidden_states = hidden_states + residual_x
            
        return hidden_states

# Example usage:
def example_usage():
    # Create a sample input tensor
    batch_size, seq_length, input_dim = 1, 4086, 256
   
    output_dim = 128  # Reduced output dimension
    x = torch.randn(batch_size, seq_length, input_dim)
    
    res_x = torch.randn(1,4086,128)
    # Initialize adapter
    adapter = ConditionedResidualAdapterBottleneck(
        input_dim=input_dim,
        output_dim=output_dim,
        bottleneck_dim=64,
        adapter_dropout=0.1,
        adapter_init_scale=1e-3
    )
    adapter.requires_grad_(True)

    # Forward pass
    output = adapter(residual_x=res_x,conditioned_x=x)
    print(f"Input shape: {x.shape}")  
    print(f"Output shape: {output.shape}") 

if __name__ == "__main__":
    example_usage()