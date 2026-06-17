from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from .dts_aux import (
    get_ranks_from_fitness_values,
    get_reward_from_fitness_scores,
    get_trajectory_probability_from_log_probs,
)
from .tournament_utils import get_fit, tournament_selection
from .population_to_vec_transformer import PopulationToVecTransformer
from .self_attention_pointer import SelfAttentionPointer


def get_tournament_indexes(population_, n_to_select_, fitness_dict_, tournament_size=5):
    return tournament_selection(
        population_, n_to_select_, tournament_size, fitness_dict_, return_index=True
    )


class DTSPolicy:
    """Core engine of Deep Tournament Selection (DTS): a Transformer encoder + a
    self-attention pointer, trained online with REINFORCE."""

    def __init__(
        self,
        pop_to_vec_transformer: PopulationToVecTransformer,
        pointer_transformer: SelfAttentionPointer,
        device="cpu",
        train_every_n_gens=10,
        learning_rate=2e-3,
        normalize_reward_batches=True,
        adam_decay=0,
        clip_grad_norm=1.0,
        clear_fitness_dict_after_sampling=True,
        epsilon_greedy=1.0,
        epsilon_greedy_decay=0.999,
        min_epsilon=0.2,
        teacher_forcing_algorithm=None,
        custom_reward_function=None,
        amsgrad=True,
        center_only=False,
        final_lr=None,
    ):

        assert device in ["cpu", "cuda"], "device must be either 'cpu' or 'cuda'"
        assert train_every_n_gens > 0, "train_every_n_gens must be greater than 0"
        assert clip_grad_norm is None or clip_grad_norm > 0, (
            "clip_grad_norm must be None or greater than 0"
        )
        if final_lr is not None:
            assert final_lr < learning_rate, "final_lr must be less than learning_rate"

        if teacher_forcing_algorithm is None:
            teacher_forcing_algorithm = get_tournament_indexes

        if custom_reward_function is None:
            custom_reward_function = get_reward_from_fitness_scores

        self.pop_to_vec_transformer = pop_to_vec_transformer.to(device)
        self.pointer_transformer = pointer_transformer.to(device)
        self.all_parameters = list(self.pop_to_vec_transformer.parameters()) + list(
            self.pointer_transformer.parameters()
        )
        self.optimizer = torch.optim.Adam(
            self.all_parameters,
            lr=learning_rate,
            weight_decay=adam_decay,
            amsgrad=amsgrad,
        )

        if final_lr is not None:
            total_epochs = 6_000 / train_every_n_gens

            def lr_lambda(epoch):
                frac = min(epoch / total_epochs, 1.0)
                return (1 - frac) + frac * (final_lr / learning_rate)

            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda=lr_lambda
            )

        else:
            self.scheduler = None

        self.learning_rate: float = learning_rate
        self.device: str = device
        self.train_every_n_gens: int = train_every_n_gens
        self.normalize_reward_batches: bool = normalize_reward_batches
        self.clip_grad_norm: float = clip_grad_norm
        self.teacher_forcing_algorithm = teacher_forcing_algorithm
        self.clear_fitness_dict_after_sampling: bool = clear_fitness_dict_after_sampling

        self.epsilon_greedy: float = epsilon_greedy
        self.epsilon_greedy_decay: float = epsilon_greedy_decay
        self.min_epsilon: float = min_epsilon
        self.center_only = center_only

        self.custom_reward_function = custom_reward_function
        self.pending_trajectories_log_probs = []
        self.trajectory_probabilities = []
        self.trajectory_rewards = []
        self.generation_tracked_fitness_values = []

    def select(
        self,
        population: np.ndarray,
        n_to_select: int,
        fitness_dict: dict,
        generation_index: int,
    ) -> np.ndarray:
        """
        fitness is assumed to be maximized.
        This function is expected to be called at each generation.
        """

        fitness_values = [get_fit(ind, fitness_dict) for ind in population]

        self.save_trajectory_from_previous_gen(
            fitness_values, generation_index, population
        )

        self.run_epoch_check(fitness_dict, fitness_values, population)

        (
            trajectory_log_probabilities,
            next_gen_population,
            selected_population_indices,
        ) = self.predict_deep_tournament_selection(
            population, fitness_values, n_to_select, fitness_dict
        )

        total_log_prob = (
            get_trajectory_probability_from_log_probs(
                trajectory_log_probabilities, selected_population_indices
            )
            .sum()
            .unsqueeze(0)
        )

        self.pending_trajectories_log_probs.append(total_log_prob)

        self.epsilon_greedy = max(
            self.epsilon_greedy * self.epsilon_greedy_decay, self.min_epsilon
        )
        return next_gen_population

    def run_epoch_check(
        self, fitness_dict: dict, fitness_values: list, population: np.ndarray
    ):
        if self.get_current_batch_size() >= self.train_every_n_gens:
            self.run_epoch()
            if self.clear_fitness_dict_after_sampling:
                fitness_dict.clear()
                for ind, fit in zip(population, fitness_values):
                    fitness_dict[tuple(ind)] = fit

    def predict_deep_tournament_selection(
        self,
        population: np.ndarray,
        fitness_values: list[float],
        n_to_select: int,
        fitness_dict: dict,
    ):

        fitness_values = np.array(fitness_values)

        tournament_selection_prediction_indexes, all_tournaments_indexes = (
            self.teacher_forcing_algorithm(population, n_to_select, fitness_dict)
        )

        populations_tensor = torch.tensor(
            population, device=self.device, dtype=torch.long
        )

        population_order = torch.tensor(
            get_ranks_from_fitness_values(fitness_values.reshape(1, -1)),
            device=self.device,
            dtype=torch.long,
        )

        embedded_population = self.pop_to_vec_transformer(
            populations_tensor.unsqueeze(0), population_order
        ).squeeze(0)

        indices_per_tournament = np.argmax(
            all_tournaments_indexes == tournament_selection_prediction_indexes[:, None],
            axis=1,
        )
        teacher_forcing_indexes = torch.tensor(
            indices_per_tournament, device=self.device, dtype=torch.long
        ).view(-1, 1)

        tournament_populations = populations_tensor[all_tournaments_indexes]
        tournament_populations_embeddings = embedded_population[all_tournaments_indexes]
        fitness_values_per_tournament = fitness_values[all_tournaments_indexes]

        trajectory_log_probabilities, selected_population_indices = (
            self.predict_batch_selection(
                tournament_populations,
                fitness_values_per_tournament,
                n_to_select=1,
                teacher_forcing_indexes=teacher_forcing_indexes,
                embedded_population=tournament_populations_embeddings,
            )
        )

        selected_population_after_tournament = (
            tournament_populations[
                torch.arange(tournament_populations.shape[0]),
                selected_population_indices.squeeze(1),
            ]
            .cpu()
            .numpy()
        )

        return (
            trajectory_log_probabilities,
            selected_population_after_tournament,
            selected_population_indices,
        )

    def predict_batch_selection(
        self,
        population: np.ndarray | Tensor,
        fitness_values: np.ndarray,
        n_to_select: int,
        teacher_forcing_indexes,
        embedded_population=None,
    ):
        if embedded_population is None:
            populations_tensor = torch.tensor(
                population, device=self.device, dtype=torch.long
            )
            embedded_population = self.pop_to_vec_transformer(populations_tensor)

        population_order = torch.tensor(
            get_ranks_from_fitness_values(fitness_values),
            device=self.device,
            dtype=torch.long,
        )

        if teacher_forcing_indexes is not None:
            teacher_forcing_indexes = torch.as_tensor(
                teacher_forcing_indexes, device=self.device, dtype=torch.long
            )

        trajectory_log_probabilities, selected_population_indices = (
            self.pointer_transformer(
                embedded_population,
                n_to_select,
                teacher_forcing_indexes,
                population_order,
                self.epsilon_greedy,
            )
        )
        return trajectory_log_probabilities, selected_population_indices

    def save_trajectory_from_previous_gen(
        self, fitness_values, generation_index: int, population: np.ndarray
    ):
        self.generation_tracked_fitness_values.append(np.array(fitness_values))

        if generation_index >= 1:
            cur_gen_fitness_metric = self.generation_tracked_fitness_values[
                generation_index
            ]
            prev_gen_fitness_metric = self.generation_tracked_fitness_values[
                generation_index - 1
            ]
            reward = self.custom_reward_function(
                cur_gen_fitness_metric, prev_gen_fitness_metric, population
            )
            self.log_trajectory_to_memory(
                self.pending_trajectories_log_probs.pop(0), rewards=[reward]
            )

    def get_loss(self, all_rewards, all_traj_proba, numerical_stability=1e-10):
        if self.normalize_reward_batches:
            advantages = all_rewards - torch.mean(all_rewards)
            if not self.center_only:
                advantages = advantages / (torch.std(all_rewards) + numerical_stability)

        else:
            advantages = all_rewards

        advantages = torch.clamp(advantages, -5, 5)
        advantages = advantages.to(self.device)

        loss = -torch.mean(all_traj_proba * advantages).to(self.device)
        return loss

    def run_epoch(self, numerical_stability=1e-10):
        all_traj_proba = torch.cat(self.trajectory_probabilities).to(self.device)
        all_rewards = torch.cat(self.trajectory_rewards).to(self.device)

        self.trajectory_probabilities.clear()
        self.trajectory_rewards.clear()
        self.optimizer.zero_grad()

        loss = self.get_loss(all_rewards, all_traj_proba, numerical_stability)

        loss.backward()
        loss_value = loss.item()

        if self.clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.all_parameters, self.clip_grad_norm)

        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        print(f"loss: {loss_value}, reward: {torch.mean(all_rewards)}")

    def log_trajectory_to_memory(self, trajectory_proba, rewards):
        self.trajectory_probabilities.append(trajectory_proba)
        self.trajectory_rewards.append(torch.tensor(rewards, device=self.device))

    def get_current_batch_size(self):
        return sum([len(rewards) for rewards in self.trajectory_rewards])
