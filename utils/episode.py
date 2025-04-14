import torch
from torch.utils.data import Sampler
import random
from collections import defaultdict

class FewShotEpisodeSampler(Sampler):
    def __init__(self, dataset, num_episodes, n_way, k_shot, q_query):
        """
        Args:
            dataset: torch Dataset with `.labels` or `.targets`
            num_episodes: number of episodes per epoch
            n_way: number of classes per episode
            k_shot: number of support examples per class
            q_query: number of query examples per class
        """
        self.dataset = dataset
        self.num_episodes = num_episodes
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query

        # Collect indices for each class
        self.class_to_indices = defaultdict(list)
        for idx, label in enumerate(dataset.labels if hasattr(dataset, "labels") else dataset.targets):
            self.class_to_indices[label].append(idx)
        self.classes = list(self.class_to_indices.keys())

    def __len__(self):
        return self.num_episodes

    def __iter__(self):
        for _ in range(self.num_episodes):
            selected_classes = random.sample(self.classes, self.n_way)
            episode_indices = []

            for class_idx in selected_classes:
                indices = random.sample(self.class_to_indices[class_idx], self.k_shot + self.q_query)
                episode_indices.extend(indices)

            yield episode_indices
