from __future__ import annotations

import math

import torch


class ActorCritic(torch.nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: tuple[int, int] | list[int] = (256, 256)) -> None:
        super().__init__()
        sizes = tuple(hidden_sizes)
        if len(sizes) != 2:
            raise ValueError("hidden_sizes must contain two integers")
        self.policy_net = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, sizes[0]),
            torch.nn.Tanh(),
            torch.nn.Linear(sizes[0], sizes[1]),
            torch.nn.Tanh(),
        )
        self.policy_head = torch.nn.Linear(sizes[1], action_dim)
        self.value_net = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, sizes[0]),
            torch.nn.Tanh(),
            torch.nn.Linear(sizes[0], sizes[1]),
            torch.nn.Tanh(),
        )
        self.value_head = torch.nn.Linear(sizes[1], 1)
        self.log_std = torch.nn.Parameter(torch.zeros(action_dim))
        self._init_weights()

    def _init_weights(self) -> None:
        for layer in self.policy_net:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.orthogonal_(layer.weight, gain=math.sqrt(2))
                torch.nn.init.constant_(layer.bias, 0.0)
        for layer in self.value_net:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.orthogonal_(layer.weight, gain=math.sqrt(2))
                torch.nn.init.constant_(layer.bias, 0.0)
        torch.nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        torch.nn.init.constant_(self.policy_head.bias, 0.0)
        torch.nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        torch.nn.init.constant_(self.value_head.bias, 0.0)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.policy_head(self.policy_net(obs))
        value = self.value_head(self.value_net(obs)).squeeze(-1)
        return mean, value

    def get_dist_and_value(self, obs: torch.Tensor) -> tuple[torch.distributions.Normal, torch.Tensor]:
        mean, value = self.forward(obs)
        std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(mean, std)
        return dist, value
