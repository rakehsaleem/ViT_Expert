"""
Dataset Classes for ViT Expert Distillation

This module implements dataset classes for loading multimodal data including
video frames, gaze history, and ground truth saliency maps.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import os
import json
from typing import Dict, List, Tuple, Optional
from PIL import Image
import torchvision.transforms as transforms


class MultimodalSaliencyDataset(Dataset):
    """
    Dataset class for multimodal saliency prediction.
    
    Loads video frames, gaze history, and ground truth saliency maps
    for training the teacher and student models.
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        img_size: int = 224,
        gaze_seq_len: int = 10,
        transform: Optional[transforms.Compose] = None
    ):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Root directory containing processed data
            split: Dataset split ('train', 'val', 'test')
            img_size: Target image size for resizing
            gaze_seq_len: Length of gaze history sequence
            transform: Optional image transformations
        """
        self.data_dir = data_dir
        self.split = split
        self.img_size = img_size
        self.gaze_seq_len = gaze_seq_len
        
        # Set up image transforms
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
        
        # Load data samples
        self.samples = self._load_samples()
        
        print(f"Loaded {len(self.samples)} samples for {split} split")
    
    def _load_samples(self) -> List[Dict]:
        """
        Load sample metadata from the data directory.
        
        Returns:
            List of sample dictionaries with file paths
        """
        samples = []
        
        # Define paths
        video_dir = os.path.join(self.data_dir, 'processed', 'video_features')
        gaze_dir = os.path.join(self.data_dir, 'processed', 'gaze_features')
        saliency_dir = os.path.join(self.data_dir, 'processed', 'gt_saliency_maps')
        
        # Check if directories exist
        if not all(os.path.exists(d) for d in [video_dir, gaze_dir, saliency_dir]):
            print("Warning: Data directories not found. Creating dummy samples for testing.")
            return self._create_dummy_samples()
        
        # Load sample files
        video_files = sorted([f for f in os.listdir(video_dir) if f.endswith('.npy')])
        
        for video_file in video_files:
            sample_id = video_file.replace('.npy', '')
            
            gaze_file = os.path.join(gaze_dir, f"{sample_id}_gaze.json")
            saliency_file = os.path.join(saliency_dir, f"{sample_id}_saliency.npy")
            
            if os.path.exists(gaze_file) and os.path.exists(saliency_file):
                samples.append({
                    'sample_id': sample_id,
                    'video_path': os.path.join(video_dir, video_file),
                    'gaze_path': gaze_file,
                    'saliency_path': saliency_file
                })
        
        return samples
    
    def _create_dummy_samples(self, num_samples: int = 100) -> List[Dict]:
        """
        Create dummy samples for testing when real data is not available.
        
        Args:
            num_samples: Number of dummy samples to create
            
        Returns:
            List of dummy sample dictionaries
        """
        samples = []
        
        for i in range(num_samples):
            samples.append({
                'sample_id': f'dummy_{i:04d}',
                'video_path': f'dummy_video_{i:04d}.npy',
                'gaze_path': f'dummy_gaze_{i:04d}.json',
                'saliency_path': f'dummy_saliency_{i:04d}.npy'
            })
        
        return samples
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing video_frame, gaze_history, and ground_truth
        """
        sample = self.samples[idx]
        
        # Load video frame
        video_frame = self._load_video_frame(sample['video_path'])
        
        # Load gaze history
        gaze_history = self._load_gaze_history(sample['gaze_path'])
        
        # Load ground truth saliency
        ground_truth = self._load_saliency_map(sample['saliency_path'])
        
        return {
            'video_frame': video_frame,
            'gaze_history': gaze_history,
            'ground_truth': ground_truth,
            'sample_id': sample['sample_id']
        }
    
    def _load_video_frame(self, video_path: str) -> torch.Tensor:
        """
        Load and preprocess a video frame.
        
        Args:
            video_path: Path to video frame file
            
        Returns:
            Preprocessed video frame tensor
        """
        if video_path.startswith('dummy_'):
            # Create dummy video frame
            dummy_frame = torch.randn(3, self.img_size, self.img_size)
            return dummy_frame
        
        try:
            # Load video frame (assuming it's saved as numpy array)
            frame = np.load(video_path)
            
            # Convert to PIL Image if needed
            if frame.shape[-1] == 3:  # RGB
                frame = Image.fromarray(frame.astype(np.uint8))
            else:  # Grayscale
                frame = Image.fromarray(frame.astype(np.uint8)).convert('RGB')
            
            # Apply transforms
            frame = self.transform(frame)
            return frame
            
        except Exception as e:
            print(f"Error loading video frame {video_path}: {e}")
            # Return dummy frame on error
            return torch.randn(3, self.img_size, self.img_size)
    
    def _load_gaze_history(self, gaze_path: str) -> torch.Tensor:
        """
        Load gaze history sequence.
        
        Args:
            gaze_path: Path to gaze data file
            
        Returns:
            Gaze history tensor [seq_len, 2]
        """
        if gaze_path.startswith('dummy_'):
            # Create dummy gaze history
            gaze_history = torch.randn(self.gaze_seq_len, 2)
            # Normalize to [0, 1] range
            gaze_history = torch.sigmoid(gaze_history)
            return gaze_history
        
        try:
            # Load gaze data (assuming JSON format)
            with open(gaze_path, 'r') as f:
                gaze_data = json.load(f)
            
            # Extract gaze coordinates
            gaze_coords = gaze_data.get('gaze_coordinates', [])
            
            if len(gaze_coords) == 0:
                # Create dummy gaze if no data
                gaze_history = torch.randn(self.gaze_seq_len, 2)
                gaze_history = torch.sigmoid(gaze_history)
                return gaze_history
            
            # Convert to tensor and pad/truncate to desired length
            gaze_tensor = torch.tensor(gaze_coords, dtype=torch.float32)
            
            if gaze_tensor.size(0) > self.gaze_seq_len:
                gaze_tensor = gaze_tensor[:self.gaze_seq_len]
            elif gaze_tensor.size(0) < self.gaze_seq_len:
                # Pad with last known gaze position
                last_gaze = gaze_tensor[-1:] if gaze_tensor.size(0) > 0 else torch.tensor([[0.5, 0.5]])
                padding = last_gaze.repeat(self.gaze_seq_len - gaze_tensor.size(0), 1)
                gaze_tensor = torch.cat([gaze_tensor, padding], dim=0)
            
            return gaze_tensor
            
        except Exception as e:
            print(f"Error loading gaze data {gaze_path}: {e}")
            # Return dummy gaze on error
            gaze_history = torch.randn(self.gaze_seq_len, 2)
            gaze_history = torch.sigmoid(gaze_history)
            return gaze_history
    
    def _load_saliency_map(self, saliency_path: str) -> torch.Tensor:
        """
        Load ground truth saliency map.
        
        Args:
            saliency_path: Path to saliency map file
            
        Returns:
            Saliency map tensor [1, H, W]
        """
        if saliency_path.startswith('dummy_'):
            # Create dummy saliency map
            saliency_map = torch.rand(1, self.img_size, self.img_size)
            return saliency_map
        
        try:
            # Load saliency map (assuming numpy format)
            saliency_map = np.load(saliency_path)
            
            # Convert to tensor
            if len(saliency_map.shape) == 2:  # 2D array
                saliency_map = torch.from_numpy(saliency_map).unsqueeze(0)
            else:  # Already 3D
                saliency_map = torch.from_numpy(saliency_map)
            
            # Resize to target size
            saliency_map = torch.nn.functional.interpolate(
                saliency_map.unsqueeze(0),
                size=(self.img_size, self.img_size),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
            
            # Normalize to [0, 1]
            saliency_map = torch.clamp(saliency_map, 0, 1)
            
            return saliency_map
            
        except Exception as e:
            print(f"Error loading saliency map {saliency_path}: {e}")
            # Return dummy saliency on error
            return torch.rand(1, self.img_size, self.img_size)


def create_data_loaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    img_size: int = 224,
    gaze_seq_len: int = 10
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders.
    
    Args:
        data_dir: Root directory containing processed data
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes
        img_size: Target image size
        gaze_seq_len: Length of gaze history sequence
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = MultimodalSaliencyDataset(
        data_dir=data_dir,
        split='train',
        img_size=img_size,
        gaze_seq_len=gaze_seq_len,
        transform=train_transform
    )
    
    val_dataset = MultimodalSaliencyDataset(
        data_dir=data_dir,
        split='val',
        img_size=img_size,
        gaze_seq_len=gaze_seq_len,
        transform=val_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, val_loader


# Example usage and testing
if __name__ == "__main__":
    # Test dataset creation
    dataset = MultimodalSaliencyDataset(
        data_dir='01_data',
        split='train',
        img_size=224,
        gaze_seq_len=10
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test data loading
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Video frame shape: {sample['video_frame'].shape}")
        print(f"Gaze history shape: {sample['gaze_history'].shape}")
        print(f"Ground truth shape: {sample['ground_truth'].shape}")
        print(f"Sample ID: {sample['sample_id']}")
    
    # Test data loaders
    train_loader, val_loader = create_data_loaders(
        data_dir='01_data',
        batch_size=4,
        num_workers=0  # Use 0 for Windows compatibility
    )
    
    print(f"Train loader batches: {len(train_loader)}")
    print(f"Val loader batches: {len(val_loader)}")
    
    # Test batch loading
    for batch in train_loader:
        print(f"Batch video shape: {batch['video_frame'].shape}")
        print(f"Batch gaze shape: {batch['gaze_history'].shape}")
        print(f"Batch saliency shape: {batch['ground_truth'].shape}")
        break
