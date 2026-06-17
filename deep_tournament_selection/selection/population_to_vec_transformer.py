import torch
from torch import nn


class PopulationToVecTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        emb_dim,
        latent_dim,
        n_heads=4,
        n_layers=2,
        dim_feedforward=256,
        max_pointers=128,
        max_gene_positions=4096,
        dropout=0.1,
        use_rank_embeddings=True,
        canonicalize=False,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, emb_dim)
        self.max_gene_positions = max_gene_positions
        self.emb_dim = emb_dim
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=n_heads,
            batch_first=True,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.proj = nn.Linear(emb_dim, latent_dim)
        self.rank_pos_emb = nn.Embedding(max_pointers, latent_dim)
        self.pool_attn = nn.Linear(emb_dim, 1, bias=False)
        self.use_rank_embeddings = use_rank_embeddings
        self.canonicalize = canonicalize

    def forward(self, x, population_order=None):
        """
        Input:
            x : LongTensor [batch, n_sentences, sentence_length]
        Output:
            embeddings : [batch, n_sentences, latent_dim]
        """

        b, n_sent, seq_len = x.shape

        x = x.reshape(b * n_sent, seq_len)

        if self.canonicalize:
            x = self.canonicalize_sequences(x)

        tok = self.token_emb(x)
        if seq_len > self.max_gene_positions:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_gene_positions={self.max_gene_positions}"
            )
        tok = tok + self.get_sinusoidal_positions(
            seq_len, x.device, tok.dtype
        ).unsqueeze(0)

        h = tok

        h = self.encoder(h)

        pool_logits = self.pool_attn(h).squeeze(-1)
        pool_weights = pool_logits.softmax(dim=1)
        pooled = torch.sum(h * pool_weights.unsqueeze(-1), dim=1)

        out = self.proj(pooled)

        out_embedding = out.reshape(b, n_sent, -1)

        if population_order is not None and self.use_rank_embeddings:
            ranked_based_embeddings = self.rank_pos_emb(population_order)
            out_embedding = out_embedding + ranked_based_embeddings

        return out_embedding

    def canonicalize_sequences(self, x):
        canonical_sequences = [self.canonicalize_single_sequence(seq) for seq in x]
        return torch.stack(canonical_sequences, dim=0)

    def canonicalize_single_sequence(self, seq):
        forward = self.get_canonical_rotation(seq)
        backward = self.get_canonical_rotation(torch.flip(seq, dims=(0,)))
        return (
            forward
            if self.is_lexicographically_smaller(forward, backward)
            else backward
        )

    def get_canonical_rotation(self, seq):
        min_value = torch.min(seq)
        candidate_starts = torch.nonzero(seq == min_value, as_tuple=False).flatten()
        best_rotation = None

        for start in candidate_starts.tolist():
            rotated = torch.roll(seq, shifts=-start, dims=0)
            if best_rotation is None or self.is_lexicographically_smaller(
                rotated, best_rotation
            ):
                best_rotation = rotated

        return best_rotation

    @staticmethod
    def is_lexicographically_smaller(left, right):
        diff_indices = torch.nonzero(left != right, as_tuple=False)
        if diff_indices.numel() == 0:
            return True

        first_diff_index = diff_indices[0, 0]
        return bool(left[first_diff_index] < right[first_diff_index])

    def get_sinusoidal_positions(self, seq_len, device, dtype):
        positions = torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(
            1
        )
        div_term = torch.exp(
            torch.arange(0, self.emb_dim, 2, device=device, dtype=torch.float32)
            * (
                -torch.log(torch.tensor(10000.0, device=device, dtype=torch.float32))
                / self.emb_dim
            )
        )

        pos_encoding = torch.zeros(
            seq_len, self.emb_dim, device=device, dtype=torch.float32
        )
        pos_encoding[:, 0::2] = torch.sin(positions * div_term)
        pos_encoding[:, 1::2] = torch.cos(
            positions * div_term[: pos_encoding[:, 1::2].shape[1]]
        )
        return pos_encoding.to(dtype=dtype)
