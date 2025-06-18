import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from gym import spaces

class ImageOnlyExtractor(nn.Module):
    def __init__(self, n_frames=8, features_dim=64):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(n_frames, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        with torch.no_grad():
            dummy = torch.zeros(1, n_frames, 84, 84)
            n_flatten = self.cnn(dummy).shape[1]

        self.output_dim = features_dim
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        features = self.cnn(x)
        return self.linear(features)

class SupervisedPolicy(nn.Module):
    def __init__(self, feature_extractor: ImageOnlyExtractor):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.mlp_head = nn.Sequential(
            nn.Linear(feature_extractor.output_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Tanh()
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.mlp_head(features)
    
class CNNFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim=64):
        super().__init__(observation_space, features_dim)
        self.extractor = ImageOnlyExtractor(n_frames=8, features_dim=features_dim)

    def forward(self, observations):
        return self.extractor(observations)

class CustomCNNPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            features_extractor_class=CNNFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=64),
            **kwargs
        )