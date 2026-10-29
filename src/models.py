"""Five selected MIL models, using the existing project implementations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from torch import nn


@dataclass(frozen=True)
class MilConfig:
    instance_hidden_dim: int = 128
    embedding_dim: int = 64
    attention_hidden_dim: int = 32
    dropout: float = 0.2
    top_k: int = 8


class DecMil(torch.nn.Module):
    def __init__(self, input_channels: int, config: MilConfig) -> None:
        super().__init__()
        self.input_channels = int(input_channels)
        self.top_k = int(config.top_k)
        if self.top_k <= 0:
            raise ValueError(f"top_k must be positive, found {self.top_k}.")
        self.instance_encoder = torch.nn.Sequential(
            torch.nn.Linear(self.input_channels, config.instance_hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(config.dropout),
            torch.nn.Linear(config.instance_hidden_dim, config.embedding_dim),
            torch.nn.GELU(),
        )
        self.instance_norm = torch.nn.LayerNorm(config.embedding_dim)
        self.dropout = torch.nn.Dropout(config.dropout)
        self.attention = torch.nn.Sequential(
            torch.nn.Linear(config.embedding_dim, config.attention_hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(config.attention_hidden_dim, 1, bias=False),
        )
        self.classifier = torch.nn.Linear(config.embedding_dim * 2, 1)

    def forward(
        self,
        bags: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if bags.ndim != 3:
            raise ValueError(f"Expected [batch, instances, channels] input, found {tuple(bags.shape)!r}.")
        instance_embeddings = self.instance_encoder(bags)
        instance_embeddings = self.instance_norm(instance_embeddings)
        attention_logits = self.attention(self.dropout(instance_embeddings)).squeeze(-1)
        attention_weights = torch.softmax(attention_logits, dim=1)

        instance_count = int(instance_embeddings.shape[1])
        top_k = min(self.top_k, instance_count)
        salient_indices = torch.topk(attention_logits, k=top_k, dim=1, largest=True, sorted=True).indices
        expanded_indices = salient_indices.unsqueeze(-1).expand(-1, -1, instance_embeddings.shape[-1])
        salient_embeddings = torch.gather(instance_embeddings, dim=1, index=expanded_indices)
        salient_embedding = salient_embeddings.mean(dim=1)
        context_embedding = instance_embeddings.mean(dim=1)
        bag_embedding = torch.cat([salient_embedding, context_embedding], dim=1)
        bag_logits = self.classifier(self.dropout(bag_embedding)).squeeze(-1)
        return bag_logits, bag_embedding, attention_weights, attention_logits, salient_indices


class AttentionMil(nn.Module):
    def __init__(self, input_channels: int, config: MilConfig) -> None:
        super().__init__()
        self.input_channels = int(input_channels)
        self.instance_encoder = nn.Sequential(
            nn.Linear(self.input_channels, config.instance_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.instance_hidden_dim, config.embedding_dim),
            nn.GELU(),
        )
        self.instance_norm = nn.LayerNorm(config.embedding_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.attention = nn.Sequential(
            nn.Linear(config.embedding_dim, config.attention_hidden_dim),
            nn.Tanh(),
            nn.Linear(config.attention_hidden_dim, 1, bias=False),
        )
        self.classifier = nn.Linear(config.embedding_dim, 1)

    def forward(self, bags: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if bags.ndim != 3:
            raise ValueError(f"Expected [batch, instances, channels] input, found {tuple(bags.shape)!r}.")
        instance_embeddings = self.instance_encoder(bags)
        instance_embeddings = self.instance_norm(instance_embeddings)
        attention_logits = self.attention(self.dropout(instance_embeddings)).squeeze(-1)
        attention_weights = torch.softmax(attention_logits, dim=1)
        bag_embedding = torch.sum(attention_weights.unsqueeze(-1) * instance_embeddings, dim=1)
        bag_logits = self.classifier(self.dropout(bag_embedding)).squeeze(-1)
        return bag_logits, bag_embedding, attention_weights


class SharedEncoder(nn.Module):
    def __init__(
        self, input_channels: int, hidden_dim: int = 128,
        embedding_dim: int = 64, dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.instance_encoder = nn.Sequential(
            nn.Linear(input_channels, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
            nn.GELU(),
        )
        self.instance_norm = nn.LayerNorm(embedding_dim)

    def forward(self, bags: torch.Tensor) -> torch.Tensor:
        return self.instance_norm(self.instance_encoder(bags))


class GatedAbmil(nn.Module):
    def __init__(self, input_channels: int, config: MilConfig, attention_hidden_dim: int = 256) -> None:
        super().__init__()
        self.encoder = SharedEncoder(
            input_channels, config.instance_hidden_dim, config.embedding_dim, config.dropout
        )
        self.attn_v = nn.Linear(config.embedding_dim, attention_hidden_dim)
        self.attn_u = nn.Linear(config.embedding_dim, attention_hidden_dim)
        self.attn_w = nn.Linear(attention_hidden_dim, 1, bias=False)
        self.classifier = nn.Linear(config.embedding_dim, 2)

    def forward(self, bags: torch.Tensor) -> Dict[str, torch.Tensor]:
        instance_embeddings = self.encoder(bags)
        gated_features = (
            torch.tanh(self.attn_v(instance_embeddings))
            * torch.sigmoid(self.attn_u(instance_embeddings))
        )
        attention_logits = self.attn_w(gated_features).squeeze(-1)
        attention_weights = torch.softmax(attention_logits, dim=1)
        bag_embedding = torch.sum(attention_weights.unsqueeze(-1) * instance_embeddings, dim=1)
        bag_logits = self.classifier(bag_embedding)
        return {
            "bag_logits": bag_logits,
            "attention_weights": attention_weights,
            "bag_embedding": bag_embedding,
        }


class _ClamGatedAttention(nn.Module):
    def __init__(self, input_dim: int = 512, hidden_dim: int = 256) -> None:
        super().__init__()
        self.attn_a = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh())
        self.attn_b = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Sigmoid())
        self.attn_c = nn.Linear(hidden_dim, 1)

    def forward(self, instance_features: torch.Tensor) -> torch.Tensor:
        return self.attn_c(self.attn_a(instance_features) * self.attn_b(instance_features))


class ClamSb(nn.Module):
    def __init__(
        self,
        input_channels: int,
        config: MilConfig,
        projection_dim: int = 512,
        attention_hidden_dim: int = 256,
        projection_dropout: float = 0.25,
        k_sample: int = 8,
    ) -> None:
        super().__init__()
        self.encoder = SharedEncoder(
            input_channels, config.instance_hidden_dim, config.embedding_dim, config.dropout
        )
        self.proj = nn.Sequential(
            nn.Linear(config.embedding_dim, projection_dim),
            nn.ReLU(),
            nn.Dropout(projection_dropout),
        )
        self.attention = _ClamGatedAttention(projection_dim, attention_hidden_dim)
        self.classifier = nn.Linear(projection_dim, 2)
        self.instance_classifiers = nn.ModuleList([nn.Linear(projection_dim, 2) for _ in range(2)])
        self.k_sample = int(k_sample)
        self.instance_loss_fn = nn.CrossEntropyLoss()

    def _instance_loss(
        self, attention: torch.Tensor, patch_features: torch.Tensor, bag_labels: torch.Tensor,
    ) -> torch.Tensor:
        bag_instance_losses: List[torch.Tensor] = []
        for bag_index in range(patch_features.shape[0]):
            attention_scores = attention[bag_index]
            bag_features = patch_features[bag_index]
            sample_count = min(self.k_sample, attention_scores.shape[0])
            positive_indices = torch.topk(attention_scores, k=sample_count, dim=0).indices
            negative_indices = torch.topk(-attention_scores, k=sample_count, dim=0).indices
            instance_features = torch.cat(
                [
                    bag_features.index_select(0, positive_indices),
                    bag_features.index_select(0, negative_indices),
                ],
                dim=0,
            )
            instance_targets = torch.cat(
                [
                    torch.ones(sample_count, device=patch_features.device, dtype=torch.long),
                    torch.zeros(sample_count, device=patch_features.device, dtype=torch.long),
                ],
                dim=0,
            )
            instance_classifier = self.instance_classifiers[int(bag_labels[bag_index].item())]
            bag_instance_losses.append(
                self.instance_loss_fn(instance_classifier(instance_features), instance_targets)
            )
        return torch.stack(bag_instance_losses).mean()

    def forward(self, bags: torch.Tensor, labels: torch.Tensor | None = None) -> Dict[str, torch.Tensor]:
        instance_embeddings = self.encoder(bags)
        projected_features = self.proj(instance_embeddings)
        attention_logits = self.attention(projected_features).squeeze(-1)
        attention_weights = torch.softmax(attention_logits, dim=1)
        bag_embedding = torch.bmm(attention_weights.unsqueeze(1), projected_features).squeeze(1)
        model_outputs = {
            "bag_logits": self.classifier(bag_embedding),
            "attention_weights": attention_weights,
            "bag_embedding": bag_embedding,
        }
        if labels is not None:
            model_outputs["instance_loss"] = self._instance_loss(
                attention_weights, projected_features, labels
            )
        return model_outputs


def build_model(model_name, input_channels, model_settings):
    model_settings = dict(model_settings)
    head_settings = model_settings.pop("heads", {}).get(model_name, {})
    if model_name in {"dec-mil", "ug-dec-mil"}:
        return DecMil(input_channels, MilConfig(**{**model_settings, **head_settings}))
    if model_name == "attention-mil":
        return AttentionMil(input_channels, MilConfig(**{**model_settings, **head_settings}))
    if model_name == "abmil":
        return GatedAbmil(input_channels, MilConfig(**model_settings), **head_settings)
    if model_name == "clam-sb":
        return ClamSb(input_channels, MilConfig(**model_settings), **head_settings)
    raise ValueError(f"Unknown model: {model_name}")


def probabilities(model, model_name, bags):
    model_outputs = model(bags)
    if model_name in {"abmil", "clam-sb"}:
        bag_logits = model_outputs["bag_logits"]
        return torch.softmax(bag_logits, dim=1)[:, 1], torch.argmax(bag_logits, dim=1)
    aml_probabilities = torch.sigmoid(model_outputs[0])
    return aml_probabilities, (aml_probabilities >= 0.5).long()
