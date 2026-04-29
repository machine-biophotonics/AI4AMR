"""
PAMIL: Policy-driven Adaptive Multi-Instance Learning
Based on: CVPR 2024 - Dynamic Policy-Driven Adaptive MIL

Adaptive crop sampling using Reinforcement Learning:
- Model learns to sample crops until confident
- Uses 3x3 grid (9 positions) for crop selection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
from PIL import Image
import random


class FeatureEncoder(nn.Module):
    """Feature encoder using EfficientNet-B0 (shared backbone)"""
    def __init__(self):
        super().__init__()
        base_model = torchvision.models.efficientnet_b0(weights='IMAGENET1K_V1')
        self.backbone = nn.Sequential(
            base_model.features,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        self.feature_dim = 1280
    
    def forward(self, x):
        return self.backbone(x)


class PolicyNetwork(nn.Module):
    """
    LSTM-based policy network for adaptive crop sampling.
    Decides which position to sample next or when to stop.
    
    Action space: 0-8 for 3x3 grid positions, 9 for STOP
    """
    def __init__(self, feature_dim=1280, hidden_dim=256, num_heads=4):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_positions = 9  # 3x3 grid
        self.num_actions = self.num_positions + 1  # positions + STOP
        
        self.encoder = FeatureEncoder()
        
        # LSTM for maintaining state across steps
        self.lstm = nn.LSTMCell(feature_dim, hidden_dim)
        
        # State projection (aggregate sampled features)
        self.state_proj = nn.Linear(feature_dim, hidden_dim)
        
        # Attention for aggregating sampled crops
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=num_heads, 
            batch_first=True
        )
        
        # Output head for action probabilities
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, self.num_actions)
        )
        
        # Classifier head for final prediction
        self.classifier = nn.Linear(hidden_dim, 1)
        
        self.h_state = None
        self.c_state = None
    
    def reset_states(self, batch_size=1):
        """Reset LSTM hidden states"""
        device = next(self.parameters()).device
        self.h_state = torch.zeros(batch_size, self.hidden_dim, device=device)
        self.c_state = torch.zeros(batch_size, self.hidden_dim, device=device)
    
    def forward(self, crop_features, return_probs=False):
        """
        Args:
            crop_features: [batch_size, num_crops, feature_dim] - already extracted crops
            return_probs: if True, return action probabilities
        
        Returns:
            action_probs: [batch_size, num_actions] if return_probs=True
            Otherwise returns logits
        """
        batch_size = crop_features.shape[0]
        
        # Aggregate sampled crops with attention
        attended, _ = self.attention(
            self.h_state.unsqueeze(1), 
            crop_features, 
            crop_features
        )
        attended = attended.squeeze(1)
        
        # Combine with LSTM state
        combined = torch.cat([attended, self.h_state], dim=1)
        
        # Update LSTM
        self.h_state, self.c_state = self.lstm(crop_features.mean(dim=1), (self.h_state, self.c_state))
        
        # Get action logits
        logits = self.action_head(combined)
        
        if return_probs:
            return F.softmax(logits, dim=-1)
        return logits
    
    def get_action(self, crop_features, epsilon=0.1, deterministic=False):
        """
        Sample action from policy.
        
        Args:
            crop_features: [batch_size, num_crops, feature_dim]
            epsilon: exploration rate (epsilon-greedy)
            deterministic: if True, always take best action
        
        Returns:
            action: [batch_size] - selected action indices
            prob: [batch_size] - probability of selected action
        """
        probs = self.forward(crop_features, return_probs=True)
        
        if deterministic:
            action = probs.argmax(dim=-1)
        else:
            # Epsilon-greedy
            if random.random() < epsilon:
                # Random action (excluding already sampled positions)
                action = torch.randint(0, self.num_actions, (crop_features.shape[0],), device=probs.device)
            else:
                action = probs.multinomial(num_samples=1).squeeze(-1)
        
        prob = probs.gather(1, action.unsqueeze(-1)).squeeze(-1)
        return action, prob


class MILEnvironment:
    """
    MIL Environment for RL training.
    Simulates the adaptive crop sampling process.
    """
    def __init__(self, encoder, model=None, max_crops=9, num_positions=9):
        self.encoder = encoder
        self.model = model
        self.max_crops = max_crops
        self.num_positions = num_positions  # 3x3 = 9 positions
        
        # 3x3 grid: position indices
        # 0 1 2
        # 3 4 5
        # 6 7 8
        
        self.position_map = {
            0: (-1, -1), 1: (-1, 0), 2: (-1, 1),
            3: (0, -1),  4: (0, 0),  5: (0, 1),
            6: (1, -1),  7: (1, 0),  8: (1, 1)
        }
    
    def reset(self, images, labels, grid_positions):
        """
        Reset environment for a batch of images.
        
        Args:
            images: [batch_size, 3, H, W] - full images
            labels: [batch_size] - ground truth labels
            grid_positions: list of (left, top) positions for crops
        
        Returns:
            initial_state: dict with initial state info
        """
        self.images = images
        self.labels = labels
        self.grid_positions = grid_positions
        
        batch_size = images.shape[0]
        device = images.device
        
        self.sampled_mask = torch.zeros(batch_size, self.num_positions, dtype=torch.bool, device=device)
        self.sampled_features = torch.zeros(batch_size, 0, 1280, device=device)
        self.step_count = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        self.state = {
            'images': images,
            'features': self.sampled_features,
            'mask': self.sampled_mask,
            'steps': self.step_count,
            'labels': labels
        }
        
        return self.state
    
    def step(self, actions):
        """
        Take actions (sample new crops or stop).
        
        Args:
            actions: [batch_size] - action indices (0-8 for positions, 9 for STOP)
        
        Returns:
            rewards: [batch_size] - rewards for this step
            dones: [batch_size] - whether episode ended
            info: dict with additional info
        """
        batch_size = self.images.shape[0]
        device = self.images.device
        
        rewards = torch.zeros(batch_size, device=device)
        dones = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        # Process each sample in batch
        for i in range(batch_size):
            action = actions[i].item()
            
            if action == self.num_positions:  # STOP action
                dones[i] = True
                # Reward based on prediction correctness
                pred = self._get_current_prediction(i)
                correct = (pred == self.labels[i].item())
                rewards[i] = 1.0 if correct else -1.0
                
                # Early stopping bonus
                steps = self.step_count[i].item()
                if correct and steps < self.max_crops:
                    rewards[i] += 0.5 * (self.max_crops - steps) / self.max_crops
                
                continue
            
            # Check if position already sampled
            if self.sampled_mask[i, action]:
                rewards[i] = -0.5  # Penalty for invalid action
                continue
            
            # Sample crop at selected position
            self.sampled_mask[i, action] = True
            self.step_count[i] += 1
            
            # Extract crop feature
            crop = self._extract_crop(i, action)
            with torch.no_grad():
                feat = self.encoder(crop).unsqueeze(0)
            self.sampled_features = torch.cat([self.sampled_features, feat], dim=1)
            
            # Reward for sampling (informative regions get higher reward)
            rewards[i] = 0.1  # Base reward for sampling
            
            # Check if max crops reached
            if self.step_count[i] >= self.max_crops:
                dones[i] = True
                pred = self._get_current_prediction(i)
                correct = (pred == self.labels[i].item())
                rewards[i] = 1.0 if correct else -1.0
        
        self.state = {
            'images': self.images,
            'features': self.sampled_features,
            'mask': self.sampled_mask,
            'steps': self.step_count,
            'labels': self.labels
        }
        
        info = {'step': self.step_count.clone()}
        return rewards, dones, info
    
    def _extract_crop(self, batch_idx, position_idx):
        """Extract crop from image at given position"""
        left, top = self.grid_positions[position_idx]
        img = self.images[batch_idx]
        H, W = img.shape[1], img.shape[2]
        crop_size = 224
        
        left = max(0, min(left, W - crop_size))
        top = max(0, min(top, H - crop_size))
        
        crop = img[:, top:top+crop_size, left:left+crop_size]
        return crop.unsqueeze(0)
    
    def _get_current_prediction(self, batch_idx):
        """Get current prediction based on sampled crops"""
        if self.sampled_features.shape[1] == 0:
            return 0
        
        features = self.sampled_features[batch_idx:batch_idx+1]
        pooled = features.mean(dim=1)
        
        with torch.no_grad():
            logits = self.model(pooled) if self.model else self.classifier(pooled)
        
        return logits.argmax(dim=-1).item()
    
    def get_valid_actions(self):
        """Get valid actions for each sample in batch"""
        batch_size = self.sampled_mask.shape[0]
        device = self.sampled_mask.device
        
        valid = ~self.sampled_mask
        stop_available = self.step_count > 0
        
        valid_with_stop = torch.cat([valid, stop_available.unsqueeze(1)], dim=1)
        return valid_with_stop


class PAMILModel(nn.Module):
    """
    Full PAMIL model combining:
    - Feature encoder (EfficientNet-B0)
    - Policy network (adaptive sampling)
    - Classifier (bag-level prediction)
    """
    def __init__(self, num_classes, num_heads=4, hidden_dim=256, max_crops=9):
        super().__init__()
        self.num_classes = num_classes
        self.max_crops = max_crops
        
        # Feature encoder
        self.encoder = FeatureEncoder()
        
        # Policy network for adaptive sampling
        self.policy = PolicyNetwork(
            feature_dim=1280,
            hidden_dim=hidden_dim,
            num_heads=num_heads
        )
        
        # Bag-level classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(1280, num_classes)
        )
    
    def forward(self, crops, return_pooled=False):
        """
        Forward pass with fixed crops (non-adaptive).
        
        Args:
            crops: [batch_size, num_crops, 3, H, W] - fixed crop crops
            return_pooled: return pooled features
        
        Returns:
            logits: [batch_size, num_classes]
            pooled: [batch_size, 1280] if return_pooled=True
        """
        batch_size, num_crops = crops.shape[:2]
        
        # Encode all crops
        crops_flat = crops.view(batch_size * num_crops, *crops.shape[2:])
        features = self.encoder(crops_flat)
        features = features.view(batch_size, num_crops, -1)
        
        # Pool with mean
        pooled = features.mean(dim=1)
        
        # Classify
        logits = self.classifier(pooled)
        
        if return_pooled:
            return logits, pooled
        return logits
    
    def adaptive_forward(self, images, grid_positions, epsilon=0.1, return_trajectory=False):
        """
        Adaptive forward pass with RL-based sampling.
        
        Args:
            images: [batch_size, 3, H, W] - full images
            grid_positions: list of (left, top) positions
            epsilon: exploration rate
            return_trajectory: return sampling trajectory
        
        Returns:
            logits: [batch_size, num_classes]
            trajectory: list of (action, prob) if return_trajectory=True
        """
        batch_size = images.shape[0]
        device = images.device
        
        # Reset policy states
        self.policy.reset_states(batch_size)
        
        # Initialize with center crop (position 4)
        center_idx = 4
        center_crop = self._extract_crops_batch(images, grid_positions[center_idx])
        sampled_features = [self.encoder(center_crop)]
        sampled_positions = [center_idx]
        trajectory = [(center_idx, 1.0)]
        
        # Update policy state
        self.policy.h_state = self.encoder(center_crop).mean(dim=1)
        self.policy.c_state = torch.zeros_like(self.policy.h_state)
        
        # Adaptive sampling loop
        max_steps = self.max_crops - 1
        done = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        for step in range(max_steps):
            # Get current features
            current_features = torch.stack(sampled_features, dim=1)
            
            # Get action from policy
            action, prob = self.policy.get_action(current_features, epsilon=epsilon)
            
            # Handle stop action
            stop_action = self.num_classes  # 9 for 3x3 grid
            
            # Process each sample
            for i in range(batch_size):
                if done[i]:
                    continue
                
                act = action[i].item()
                
                if act == stop_action:
                    done[i] = True
                    continue
                
                if act not in sampled_positions:
                    crop = self._extract_crops_batch(
                        images[i:i+1], 
                        grid_positions[act]
                    )
                    feat = self.encoder(crop)
                    sampled_features.append(feat)
                    sampled_positions.append(act)
                    trajectory.append((act, prob[i].item()))
                    
                    # Update policy state
                    self.policy.h_state[i] = feat.mean(dim=1)
            
            if done.all():
                break
        
        # Final prediction
        all_features = torch.stack(sampled_features, dim=1)
        pooled = all_features.mean(dim=1)
        logits = self.classifier(pooled)
        
        if return_trajectory:
            return logits, trajectory
        return logits
    
    def _extract_crops_batch(self, images, position):
        """Extract crop at position for batch of images"""
        left, top = position
        H, W = images.shape[2], images.shape[3]
        crop_size = 224
        
        left = max(0, min(left, W - crop_size))
        top = max(0, min(top, H - crop_size))
        
        return images[:, :, top:top+crop_size, left:left+crop_size]


def collect_episode(env, model, optimizer=None, gamma=0.99):
    """
    Collect one episode for PAMIL training.
    
    Args:
        env: MILEnvironment instance
        model: PAMIL model
        optimizer: for training (if None, just collect)
        gamma: discount factor
    
    Returns:
        trajectory: list of (state, action, reward, next_state, done)
    """
    # Reset environment
    state = env.reset()
    
    trajectory = []
    discounted_rewards = []
    
    for step in range(env.max_crops):
        # Get action from policy
        features = state['features']
        action, prob = model.policy.get_action(features)
        
        # Environment step
        rewards, dones, info = env.step(action)
        
        # Store transition
        next_state = env.state
        trajectory.append({
            'state': state,
            'action': action,
            'reward': rewards,
            'next_state': next_state,
            'done': dones,
            'prob': prob
        })
        
        # Update state
        state = next_state
        
        if dones.all():
            break
    
    # Compute discounted rewards
    cumulative = torch.zeros_like(rewards)
    for t in reversed(range(len(trajectory))):
        r = trajectory[t]['reward']
        if t == len(trajectory) - 1:
            cumulative = r
        else:
            cumulative = r + gamma * cumulative
        discounted_rewards.insert(0, cumulative)
    
    return trajectory, discounted_rewards
