
import torch
import torch.nn as nn
import torch.nn.functional as F



class CrossAttentionMaskPredictorAdaptive(nn.Module):
    def __init__(self, num_channels, spatial_dim_h, spatial_dim_w):
        """
        Cross-attention based mask predictor.
        Args:
        - num_channels: Number of input channels (e.g., 128).
        - spatial_dim: Spatial dimensions of the feature maps (e.g., 32x32).
        """
        super(CrossAttentionMaskPredictorAdaptive, self).__init__()
        self.num_channels = num_channels
        self.spatial_dim_h = spatial_dim_h
        self.spatial_dim_w = spatial_dim_w

        # Linear layers for query, key, and value projections
        self.query_proj = nn.Linear(num_channels, num_channels)
        self.key_proj = nn.Linear(spatial_dim_h * spatial_dim_w, num_channels)
        self.value_proj = nn.Linear(spatial_dim_h * spatial_dim_w, num_channels)

        # Output layer to predict the mask
        self.output_layer = nn.Linear(num_channels, num_channels)
        self.sigmoid = nn.Sigmoid()  # For binary mask probabilities

        #CooperTrim adapt learn parameter
        self.threshold = nn.Parameter(torch.tensor(0.5))  # Initialize threshold at 0.5

    def forward(self, std_dev, features):
        """
        Forward pass for the model.
        Args:
        - std_dev: Tensor of shape [batch_size, num_channels] (std dev of each channel).
        - features: Tensor of shape [batch_size, num_channels, spatial_dim, spatial_dim].
        Returns:
        - mask: Predicted binary mask probabilities [batch_size, num_channels].
        """
        batch_size, num_channels, spatial_dim_h, spatial_dim_w = features.shape

        # Flatten spatial dimensions of the features: [batch_size, num_channels, spatial_dim*spatial_dim]
        features_flat = features.view(batch_size, num_channels, -1)

        # Project standard deviation (queries): [batch_size, num_channels]
        queries = self.query_proj(std_dev)  # Shape: [batch_size, num_channels]

        # Project features to keys and values: [batch_size, num_channels, num_channels]
        keys = self.key_proj(features_flat)  # Shape: [batch_size, num_channels, num_channels]
        values = self.value_proj(features_flat)  # Shape: [batch_size, num_channels, num_channels]

        # Compute attention scores: [batch_size, num_channels]
        attention_scores = torch.einsum('bn,bmn->bm', queries, keys)  # Batch matrix multiplication
        attention_scores = F.softmax(attention_scores, dim=-1)  # Normalize scores

        # Compute attention-weighted values: [batch_size, num_channels]
        context = torch.einsum('bm,bmn->bn', attention_scores, values)

        # Predict mask using the context: [batch_size, num_channels]
        mask_logits = self.output_layer(context)  # Shape: [batch_size, num_channels]
        mask = self.sigmoid(mask_logits)  # Apply sigmoid for probabilities

        #CooperTrim adapt learn parameter
        l_threshold = torch.sigmoid(self.threshold)
        mask = (mask > l_threshold).float()
        
        return mask, l_threshold  


# Example Usage
if __name__ == "__main__":
    # Example Data
    batch_size = 1
    num_channels = 128
    spatial_dim = 32

    # Random input data of shape [batch_size, 128, 32, 32]
    input_data = torch.randn(batch_size, num_channels, spatial_dim, spatial_dim)

    # Step 1: Compute Standard Deviation for Each Channel
    std_dev = input_data.std(dim=(2, 3))  # Shape: [batch_size, 128]

    # Step 2: Initialize Model
    model = CrossAttentionMaskPredictorAdaptive(num_channels=num_channels, spatial_dim=spatial_dim)

    # Step 3: Forward Pass
    predicted_mask = model(std_dev, input_data)  # Shape: [batch_size, 128]

    # Step 4: Convert Probabilities to Binary Mask
    binary_mask = (predicted_mask > 0.5).float()  # Threshold at 0.5

    # Print Results
    print("Input Data Shape:", input_data.shape)
    print("Standard Deviation Shape:", std_dev.shape)
    print("Predicted Mask Shape:", predicted_mask.shape)
    print("Binary Mask Shape:", binary_mask.shape)

    # Example Output
    print("Predicted Mask (Probabilities):", predicted_mask)
    print("Binary Mask:", binary_mask)
