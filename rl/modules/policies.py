import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, out_dim), 
        )

    def forward(self, input):
        return self.mlp(input)


class DeterministicPolicy(nn.Module):
    def __init__(self, dimo, dimg, dima, hidden_dim=256):
        super().__init__()
        self.trunk = MLP(
            in_dim=dimo+dimg,
            out_dim=dima,
            hidden_dim=hidden_dim
        )

    def forward(self, obs):
        a = self.trunk(obs)
        return torch.tanh(a)


class Critic(nn.Module):
    def __init__(self, dimo, dimg, dima, hidden_dim):
        super().__init__()
        
        self.q = MLP(
            in_dim=dimo+dimg+dima,
            out_dim=1,
            hidden_dim=hidden_dim
        )

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        q = self.q(sa)
        return q

