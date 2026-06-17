from ..selection.dts_policy import DTSPolicy
from ..selection.eckity_adapter import DeepTournamentSelection
from ..selection.population_to_vec_transformer import PopulationToVecTransformer
from ..selection.self_attention_pointer import SelfAttentionPointer


def build_dts_operator(
    population_size,
    vocab_size,
    latent_dim=32,
    emb_dim=32,
    n_heads=4,
    n_layers=2,
    dim_feedforward=256,
    tournament_size=5,
    device="cpu",
    learning_rate=2e-3,
    final_lr=1e-3,
    train_every_n_gens=10,
    epsilon_greedy=1.0,
    epsilon_greedy_decay=0.999,
    min_epsilon=0.2,
    custom_reward_function=None,
    higher_is_better=True,
):
    """Build the DTS operator as an EC-KitY ``SelectionMethod``.

    Parameters
    ----------
    population_size : int
        Subpopulation size. The encoder's rank embedding table must cover every
        individual, so ``max_pointers`` is set to this value.
    vocab_size : int
        Number of distinct gene values + 1 (e.g. 2 for a bit-vector).
    latent_dim / emb_dim / n_heads / n_layers / dim_feedforward :
        Encoder transformer dimensions. The pointer decoder's ``d_model`` is tied
        to ``latent_dim`` because it consumes the encoder's latent embeddings.
    tournament_size : int
        Tournament size used by the teacher-forcing baseline inside DTS.
    """
    encoder = PopulationToVecTransformer(
        vocab_size=vocab_size,
        emb_dim=emb_dim,
        latent_dim=latent_dim,
        n_heads=n_heads,
        n_layers=n_layers,
        dim_feedforward=dim_feedforward,
        max_pointers=population_size,
    )
    decoder = SelfAttentionPointer(
        pointer_len=population_size,
        d_model=latent_dim,
    )

    def teacher_forcing(pop, n_select, fit_dict):
        from ..selection.tournament_utils import tournament_selection

        return tournament_selection(
            pop, n_select, tournament_size, fit_dict, return_index=True
        )

    policy = DTSPolicy(
        pop_to_vec_transformer=encoder,
        pointer_transformer=decoder,
        teacher_forcing_algorithm=teacher_forcing,
        device=device,
        learning_rate=learning_rate,
        final_lr=final_lr,
        train_every_n_gens=train_every_n_gens,
        epsilon_greedy=epsilon_greedy,
        epsilon_greedy_decay=epsilon_greedy_decay,
        min_epsilon=min_epsilon,
        custom_reward_function=custom_reward_function,
    )

    return DeepTournamentSelection(policy, higher_is_better=higher_is_better)
