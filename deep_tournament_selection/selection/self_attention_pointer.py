import torch
import torch.nn as nn

from .simple_self_attention import RecombinationAttention


class SelfAttentionPointer(nn.Module):
    def __init__(
        self,
        pointer_len,
        d_model,
        log_stability=1e-12,
        max_selection_sequence_length=32,
        sampling_method="categorical",
        teacher_forcing_strategy="hard",
        use_rank_embeddings=True,
    ):

        super().__init__()
        assert teacher_forcing_strategy in ["hard", "soft"]
        assert sampling_method in ["categorical", "greedy"], (
            "Unsupported sampling method"
        )
        self.pointer_len = pointer_len
        self.d_model = d_model
        self.log_stability = log_stability
        self.sampling_method = sampling_method
        self.teacher_forcing_strategy = teacher_forcing_strategy

        self.recomb_attn = RecombinationAttention(
            embed_dim=d_model, key_dim=d_model // 4
        )
        self.pos_emb = nn.Embedding(max_selection_sequence_length, d_model)
        self.use_rank_embeddings = use_rank_embeddings

    def forward(
        self,
        sent_embeds,
        pointer_len=None,
        teacher_forcing=None,
        population_order=None,
        epsilon_greedy=0.0,
    ):
        """
        sent_embeds: [B, N, d_model]
        teacher_forcing:[B, pointer_len] or None
        population_order:[B, N]
        Returns:
        logits: [B, pointer_len, N]
        sampled_indexes:[B, pointer_len]
        """
        if pointer_len is None:
            pointer_len = self.pointer_len

        H = sent_embeds

        if population_order is not None and self.use_rank_embeddings:
            ranked_based_embeddings = self.pos_emb(population_order)
            H = H + ranked_based_embeddings

        logits = []
        sampled_idxs = []
        w_t = self.recomb_attn(H)
        log_probs = torch.log(w_t + self.log_stability)

        for t in range(pointer_len):
            logits.append(log_probs)

            if self.teacher_forcing_strategy == "hard":
                idx = self.hard_epsilon_greedy(teacher_forcing, w_t, t, epsilon_greedy)
            else:
                idx = self.soft_epsilon_greedy(teacher_forcing, w_t, t, epsilon_greedy)

            sampled_idxs.append(idx)

        logits = torch.stack(logits, dim=1)
        sampled_idxs = torch.stack(sampled_idxs, dim=1)

        return logits, sampled_idxs

    def hard_epsilon_greedy(self, teacher_forcing, w_t, t, epsilon_greedy):
        if (teacher_forcing is not None) and (torch.rand(1).item() < epsilon_greedy):
            idx = teacher_forcing[:, t]
            return idx

        return self.sample_from_attention_dist(w_t)

    def sample_from_attention_dist(self, w_t):
        if self.sampling_method == "categorical":
            idx = torch.distributions.Categorical(probs=w_t).sample()
        else:
            _, idx = torch.max(w_t, dim=1)

        return idx

    def soft_epsilon_greedy(self, teacher_forcing, w_t, t, epsilon_greedy):
        if teacher_forcing is None:
            return self.sample_from_attention_dist(w_t)

        epsilon_greedy_mask = (torch.rand(w_t.size(0)) < epsilon_greedy).to(w_t.device)
        idx_model = self.sample_from_attention_dist(w_t)
        idx = torch.where(epsilon_greedy_mask, teacher_forcing[:, t], idx_model)
        return idx
