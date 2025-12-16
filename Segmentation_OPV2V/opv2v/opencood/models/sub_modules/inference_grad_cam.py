import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os
from pathlib import Path
import torch

# Function to create directories if they don't exist
def create_directory(directory_path):
    Path(directory_path).mkdir(parents=True, exist_ok=True)

# Function to save a figure to a specified path
def save_figure(fig, path, filename):
    full_path = os.path.join(path, filename)
    fig.savefig(full_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {full_path}")

# All visualization functions (same as before but with frame_num parameter made optional)

def visualize_channel_activations(feature_tensor, selected_channels, frame_id=None):
    """Visualize activations of selected channels"""
    # Convert tensors to numpy if they are torch tensors
    if isinstance(feature_tensor, torch.Tensor):
        feature_tensor = feature_tensor.detach().cpu().numpy()
    if isinstance(selected_channels, torch.Tensor):
        selected_channels = selected_channels.detach().cpu().numpy()
    
    # Convert to list of indices if it's a binary mask
    if len(selected_channels.shape) > 1 or (len(selected_channels.shape) == 1 and selected_channels.dtype == np.bool_):
        selected_channels = np.where(selected_channels)[0]
    
    n_channels = len(selected_channels)
    rows = (n_channels + 4) // 5  # Ceiling division to determine number of rows
    cols = min(5, n_channels)  # Maximum 5 columns
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
    if rows == 1 and cols == 1:
        axes = np.array([axes])  # Make it indexable
    axes = axes.flatten()
    
    for i, channel_idx in enumerate(selected_channels):
        if i >= len(axes):
            break
        channel_data = feature_tensor[channel_idx]
        im = axes[i].imshow(channel_data, cmap='viridis')
        axes[i].set_title(f'Channel {channel_idx}')
        axes[i].axis('off')
        fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
    
    # Hide any unused subplots
    for i in range(len(selected_channels), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    title = 'Activations of Selected Channels'
    if frame_id is not None:
        title = f'Frame {frame_id}: {title}'
    plt.suptitle(title, fontsize=16)
    plt.subplots_adjust(top=0.9)
    return fig

def simple_grad_cam(feature_tensor, selected_channels, frame_id=None):
    """Create a simple Grad-CAM-like visualization for selected channels"""
    # Convert tensors to numpy if they are torch tensors
    if isinstance(feature_tensor, torch.Tensor):
        feature_tensor = feature_tensor.detach().cpu().numpy()
    if isinstance(selected_channels, torch.Tensor):
        selected_channels = selected_channels.detach().cpu().numpy()
    
    # Convert to list of indices if it's a binary mask
    if len(selected_channels.shape) > 1 or (len(selected_channels.shape) == 1 and selected_channels.dtype == np.bool_):
        selected_channels = np.where(selected_channels)[0]
    
    # Create a weighted sum of the selected channels
    selected_features = feature_tensor[selected_channels]
    # Normalize each channel
    normalized_features = np.array([
        (channel - channel.min()) / (channel.max() - channel.min() + 1e-8)
        for channel in selected_features
    ])
    
    # Take the average across selected channels
    cam = np.mean(normalized_features, axis=0)
    
    # Create a heatmap visualization
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cam, cmap='jet')
    title = 'Simplified Grad-CAM for Selected Channels'
    if frame_id is not None:
        title = f'Frame {frame_id}: {title}'
    ax.set_title(title)
    ax.axis('off')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return fig

def compare_selected_vs_nonselected(feature_tensor, selected_channels, frame_id=None):
    """Compare average activation of selected vs non-selected channels"""
    # Convert tensors to numpy if they are torch tensors
    if isinstance(feature_tensor, torch.Tensor):
        feature_tensor = feature_tensor.detach().cpu().numpy()
    if isinstance(selected_channels, torch.Tensor):
        selected_channels = selected_channels.detach().cpu().numpy()
    
    # Convert to list of indices if it's a binary mask
    if len(selected_channels.shape) > 1 or (len(selected_channels.shape) == 1 and selected_channels.dtype == np.bool_):
        selected_channels = np.where(selected_channels)[0]
    
    non_selected = np.setdiff1d(np.arange(len(feature_tensor)), selected_channels)
    
    # Calculate average activation maps
    selected_avg = np.mean(feature_tensor[selected_channels], axis=0)
    non_selected_avg = np.mean(feature_tensor[non_selected], axis=0)
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    im0 = axes[0].imshow(selected_avg, cmap='viridis')
    axes[0].set_title('Avg of Selected Channels')
    axes[0].axis('off')
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    
    im1 = axes[1].imshow(non_selected_avg, cmap='viridis')
    axes[1].set_title('Avg of Non-Selected Channels')
    axes[1].axis('off')
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Difference map
    diff = selected_avg - non_selected_avg
    im2 = axes[2].imshow(diff, cmap='coolwarm')
    axes[2].set_title('Difference (Selected - Non-Selected)')
    axes[2].axis('off')
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    
    title = 'Channel Comparison'
    if frame_id is not None:
        title = f'Frame {frame_id}: {title}'
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    return fig

def channel_correlation_analysis(feature_tensor, selected_channels, frame_id=None):
    """Analyze correlation between selected channels"""
    # Convert tensors to numpy if they are torch tensors
    if isinstance(feature_tensor, torch.Tensor):
        feature_tensor = feature_tensor.detach().cpu().numpy()
    if isinstance(selected_channels, torch.Tensor):
        selected_channels = selected_channels.detach().cpu().numpy()
    
    # Convert to list of indices if it's a binary mask
    if len(selected_channels.shape) > 1 or (len(selected_channels.shape) == 1 and selected_channels.dtype == np.bool_):
        selected_channels = np.where(selected_channels)[0]
    
    # Reshape each channel to a 1D vector
    flattened_channels = [feature_tensor[i].flatten() for i in selected_channels]
    correlation_matrix = np.corrcoef(flattened_channels)
    
    # Visualization
    fig, ax = plt.subplots(figsize=(10, 8))
    im = sns.heatmap(
        correlation_matrix, 
        annot=True, 
        fmt=".2f", 
        cmap='coolwarm',
        xticklabels=[f'Ch {i}' for i in selected_channels],
        yticklabels=[f'Ch {i}' for i in selected_channels],
        ax=ax
    )
    title = 'Correlation Between Selected Channels'
    if frame_id is not None:
        title = f'Frame {frame_id}: {title}'
    ax.set_title(title)
    plt.tight_layout()
    return fig

def visualize_channel_importance(feature_tensor, selected_channels, frame_id=None):
    """Visualize the importance of selected channels based on activation statistics"""
    # Convert tensors to numpy if they are torch tensors
    if isinstance(feature_tensor, torch.Tensor):
        feature_tensor = feature_tensor.detach().cpu().numpy()
    if isinstance(selected_channels, torch.Tensor):
        selected_channels = selected_channels.detach().cpu().numpy()
    
    # Convert to list of indices if it's a binary mask
    if len(selected_channels.shape) > 1 or (len(selected_channels.shape) == 1 and selected_channels.dtype == np.bool_):
        selected_channels = np.where(selected_channels)[0]
    
    # Calculate channel statistics
    channel_mean = np.mean(feature_tensor, axis=(1, 2))
    channel_std = np.std(feature_tensor, axis=(1, 2))
    channel_max = np.max(feature_tensor, axis=(1, 2))
    
    # Create a figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    
    # Plot mean activation for all channels, highlighting selected ones
    axes[0].bar(range(len(feature_tensor)), channel_mean, alpha=0.5, color='gray')
    axes[0].bar(selected_channels, channel_mean[selected_channels], color='red')
    axes[0].set_title('Mean Activation by Channel')
    axes[0].set_xlabel('Channel Index')
    axes[0].set_ylabel('Mean Activation')
    
    # Plot standard deviation for all channels, highlighting selected ones
    axes[1].bar(range(len(feature_tensor)), channel_std, alpha=0.5, color='gray')
    axes[1].bar(selected_channels, channel_std[selected_channels], color='green')
    axes[1].set_title('Activation Standard Deviation by Channel')
    axes[1].set_xlabel('Channel Index')
    axes[1].set_ylabel('Standard Deviation')
    
    # Plot max activation for all channels, highlighting selected ones
    axes[2].bar(range(len(feature_tensor)), channel_max, alpha=0.5, color='gray')
    axes[2].bar(selected_channels, channel_max[selected_channels], color='blue')
    axes[2].set_title('Maximum Activation by Channel')
    axes[2].set_xlabel('Channel Index')
    axes[2].set_ylabel('Max Activation')
    
    title = 'Channel Importance Metrics'
    if frame_id is not None:
        title = f'Frame {frame_id}: {title}'
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    return fig

def visualize_channel_embedding(feature_tensor, selected_channels, frame_id=None):
    """Visualize channels in a lower-dimensional space using PCA and t-SNE"""
    # Convert tensors to numpy if they are torch tensors
    if isinstance(feature_tensor, torch.Tensor):
        feature_tensor = feature_tensor.detach().cpu().numpy()
    if isinstance(selected_channels, torch.Tensor):
        selected_channels = selected_channels.detach().cpu().numpy()
    
    # Convert to list of indices if it's a binary mask
    if len(selected_channels.shape) > 1 or (len(selected_channels.shape) == 1 and selected_channels.dtype == np.bool_):
        selected_channels = np.where(selected_channels)[0]
    
    # Reshape each channel to a 1D vector
    flattened_channels = [feature_tensor[i].flatten() for i in range(len(feature_tensor))]
    
    # Apply PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(flattened_channels)
    
    # Apply t-SNE (can be slow for large datasets)
    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(flattened_channels)
    
    # Create a figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot PCA
    scatter1 = axes[0].scatter(
        pca_result[:, 0], 
        pca_result[:, 1],
        c=np.arange(len(feature_tensor)),
        cmap='viridis',
        alpha=0.7
    )
    axes[0].scatter(
        pca_result[selected_channels, 0],
        pca_result[selected_channels, 1],
        c='red',
        s=100,
        marker='*',
        label='Selected'
    )
    axes[0].set_title('PCA of Channel Features')
    axes[0].set_xlabel('Principal Component 1')
    axes[0].set_ylabel('Principal Component 2')
    axes[0].legend()
    
    # Plot t-SNE
    scatter2 = axes[1].scatter(
        tsne_result[:, 0], 
        tsne_result[:, 1],
        c=np.arange(len(feature_tensor)),
        cmap='viridis',
        alpha=0.7
    )
    axes[1].scatter(
        tsne_result[selected_channels, 0],
        tsne_result[selected_channels, 1],
        c='red',
        s=100,
        marker='*',
        label='Selected'
    )
    axes[1].set_title('t-SNE of Channel Features')
    axes[1].set_xlabel('t-SNE Component 1')
    axes[1].set_ylabel('t-SNE Component 2')
    axes[1].legend()
    
    plt.colorbar(scatter1, ax=axes[0], label='Channel Index')
    plt.colorbar(scatter2, ax=axes[1], label='Channel Index')
    
    title = 'Channel Embedding Visualization'
    if frame_id is not None:
        title = f'Frame {frame_id}: {title}'
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    return fig

def visualize_individual_channels(feature_tensor, selected_channels, frame_id=None):
    """Create detailed visualizations for each selected channel"""
    # Convert tensors to numpy if they are torch tensors
    print("shapes=",feature_tensor.shape, selected_channels.shape)
    if isinstance(feature_tensor, torch.Tensor):
        feature_tensor = feature_tensor.detach().cpu().numpy()
    if isinstance(selected_channels, torch.Tensor):
        selected_channels = selected_channels.detach().cpu().numpy()
    
    # Convert to list of indices if it's a binary mask
    if len(selected_channels.shape) > 1 or (len(selected_channels.shape) == 1 and selected_channels.dtype == np.bool_):
        selected_channels = np.where(selected_channels)[0]
    
    for i, channel_idx in enumerate(selected_channels):
        channel_data = feature_tensor[channel_idx]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original channel visualization
        im0 = axes[0].imshow(channel_data, cmap='viridis')
        axes[0].set_title(f'Channel {channel_idx} Activation')
        axes[0].axis('off')
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        
        # Histogram of activation values
        axes[1].hist(channel_data.flatten(), bins=50, color='blue', alpha=0.7)
        axes[1].set_title(f'Channel {channel_idx} Activation Distribution')
        axes[1].set_xlabel('Activation Value')
        axes[1].set_ylabel('Frequency')
        
        # Contour plot to show activation regions
        im2 = axes[2].contourf(channel_data, cmap='jet', levels=20)
        axes[2].set_title(f'Channel {channel_idx} Contour Map')
        axes[2].axis('off')
        fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
        
        title = f'Detailed Analysis of Channel {channel_idx}'
        if frame_id is not None:
            title = f'Frame {frame_id}: {title}'
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        
        # Return the figure so it can be saved by the calling function
        yield fig, f'channel_{channel_idx}_detailed.png'

# Main function to process feature tensors and selected channels
def process_feature_visualization(feature_tensor, selected_channels, output_dir, frame_id=None):
    """
    Process feature tensor and selected channels to create and save visualizations
    
    Args:
        feature_tensor: The feature tensor from your model (can be torch.Tensor or numpy array)
        selected_channels: The selected channels (can be torch.Tensor or numpy array)
        output_dir: Directory to save the visualizations
        frame_id: Optional frame identifier for labeling (default: None)
    """
    # Create output directory
    create_directory(output_dir)
    
    # If feature_tensor is a batch, take the first item
    if isinstance(feature_tensor, torch.Tensor) and len(feature_tensor.shape) > 3:
        feature_tensor = feature_tensor[0]  # Take the first item from the batch
    
    # 1. Channel Activation Visualization
    fig1 = visualize_channel_activations(feature_tensor, selected_channels, frame_id)
    save_figure(fig1, output_dir, 'channel_activations.png')
    
    # 2. Simple Grad-CAM-like visualization
    fig2 = simple_grad_cam(feature_tensor, selected_channels, frame_id)
    save_figure(fig2, output_dir, 'grad_cam.png')
    
    # 3. Feature Map Comparison
    fig3 = compare_selected_vs_nonselected(feature_tensor, selected_channels, frame_id)
    save_figure(fig3, output_dir, 'feature_comparison.png')
    
    # 4. Channel Correlation Analysis
    fig4 = channel_correlation_analysis(feature_tensor, selected_channels, frame_id)
    save_figure(fig4, output_dir, 'correlation_analysis.png')
    
    # 5. Channel Importance Visualization
    fig5 = visualize_channel_importance(feature_tensor, selected_channels, frame_id)
    save_figure(fig5, output_dir, 'channel_importance.png')
    
    # 6. Dimensionality Reduction for Channel Visualization
    # fig6 = visualize_channel_embedding(feature_tensor, selected_channels, frame_id)
    # save_figure(fig6, output_dir, 'channel_embedding.png')
    
    # 7. Individual channel detailed visualizations
    for fig, filename in visualize_individual_channels(feature_tensor, selected_channels, frame_id):
        save_figure(fig, output_dir, filename)
    
    print(f"All visualizations saved to {output_dir}")
