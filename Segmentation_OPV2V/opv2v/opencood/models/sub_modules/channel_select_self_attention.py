import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttentionMaskPredictor(nn.Module):
    def __init__(self, num_channels, spatial_dim):
        """
        Self-attention based mask predictor using learned spatial features.
        Args:
        - num_channels: Number of input channels (e.g., 128).
        - spatial_dim: Spatial dimensions of the feature maps (e.g., 32x32).
        """
        super(SelfAttentionMaskPredictor, self).__init__()
        self.num_channels = num_channels
        self.spatial_dim = spatial_dim

        # Linear layers for query, key, and value projections
        self.query_proj = nn.Linear(num_channels, num_channels)
        self.key_proj = nn.Linear(num_channels, num_channels)
        self.value_proj = nn.Linear(num_channels, num_channels)

        # Output layer to predict the mask
        self.output_layer = nn.Linear(num_channels, num_channels)
        self.sigmoid = nn.Sigmoid()  # For binary mask probabilities

    def forward(self, features):
        """
        Forward pass for the model.
        Args:
        - features: Tensor of shape [batch_size, num_channels, spatial_dim, spatial_dim].
        Returns:
        - mask: Predicted binary mask probabilities [batch_size, num_channels].
        """
        batch_size, num_channels, spatial_dim, _ = features.shape

        # Flatten spatial dimensions into a single feature vector per channel: [batch_size, num_channels, spatial_dim * spatial_dim]
        features_flat = features.view(batch_size, num_channels, -1)  # Shape: [batch_size, num_channels, spatial_dim^2]

        # Aggregate spatial information by summing across spatial dimensions: [batch_size, num_channels]
        spatial_features = features_flat.sum(dim=-1)  # Shape: [batch_size, num_channels]

        # Project spatial features to queries, keys, and values
        queries = self.query_proj(spatial_features)  # Shape: [batch_size, num_channels]
        keys = self.key_proj(spatial_features)       # Shape: [batch_size, num_channels]
        values = self.value_proj(spatial_features)   # Shape: [batch_size, num_channels]

        # Compute attention scores: [batch_size, num_channels, num_channels]
        attention_scores = torch.einsum('bq,bk->bqk', queries, keys)  # Batch matrix multiplication
        attention_scores = F.softmax(attention_scores, dim=-1)  # Normalize scores along the last dimension

        # Compute attention-weighted values: [batch_size, num_channels]
        context = torch.einsum('bqk,bk->bq', attention_scores, values)

        # Predict mask using the context: [batch_size, num_channels]
        mask_logits = self.output_layer(context)  # Shape: [batch_size, num_channels]
        mask = self.sigmoid(mask_logits)  # Apply sigmoid for probabilities

        return mask


# Example Usage
if __name__ == "__main__":
    # Example Data
    batch_size = 1
    num_channels = 128
    spatial_dim = 32

    # Random input data of shape [batch_size, 128, 32, 32]
    input_data = torch.randn(batch_size, num_channels, spatial_dim, spatial_dim)

    # Step 1: Initialize Model
    model = SelfAttentionMaskPredictor(num_channels=num_channels, spatial_dim=spatial_dim)

    # Step 2: Forward Pass
    predicted_mask = model(input_data)  # Shape: [batch_size, 128]

    # Step 3: Convert Probabilities to Binary Mask
    binary_mask = (predicted_mask > 0.5).float()  # Threshold at 0.5

    # Print Results
    print("Input Data Shape:", input_data.shape)
    print("Predicted Mask Shape:", predicted_mask.shape)
    print("Binary Mask Shape:", binary_mask.shape)

    # Example Output
    print("Predicted Mask (Probabilities):", predicted_mask)
    print("Binary Mask:", binary_mask)
