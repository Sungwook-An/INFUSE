import torch
import numpy as np


class CategoriesSampler:
    """
    CategoriesSampler for N-way K-shot learning.
    For each episode:
        - Randomly sample N classes
        - For each class, sample K + Q examples (support + query)
    """
    
    def __init__(self, labels, n_batch, n_way, n_shot, n_query):
        """
        Args:
            labels (list): List of labels for each example.
            n_batch (int): Number of batches.
            n_way (int): Number of classes per episode.
            n_shot (int): Number of examples per class in the support set.
            n_query (int): Number of examples per class in the query set.
        """
        
        self.labels = np.array(labels)
        self.n_batch = n_batch
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        
        self.class_indices = []
        for i in range(max(labels) + 1):
            indices = np.argwhere(self.labels == i).reshape(-1)
            self.class_indices.append(torch.from_numpy(indices))
            
    def __len__(self):
        return self.n_batch
    
    def __iter__(self):
        for _ in range(self.n_batch):
            # Randomly sample N classes
            classes = torch.randperm(len(self.class_indices))[:self.n_way]
            
            support_indices = []
            query_indices = []
            
            for c in classes:
                indices = self.class_indices[c]
                perm = torch.randperm(len(indices))
                selected = indices[perm[:self.n_shot + self.n_query]]
                support_indices.append(selected[:self.n_shot])
                query_indices.append(selected[self.n_shot:])
            
            support_indices = torch.stack(support_indices).view(-1)
            query_indices = torch.stack(query_indices).view(-1)
            
            yield support_indices, query_indices
            

########################################################
#############  Checking the implementation #############
########################################################
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from collections import Counter

    labels = np.random.randint(0, 10, size=1000)

    sampler = CategoriesSampler(labels, n_batch=2, n_way=5, n_shot=1, n_query=3)

    for episode_idx, (support_idx, query_idx) in enumerate(sampler):
        print(f"\n=== Episode {episode_idx} ===:")
        print(f"Support indices: {support_idx.tolist()}")
        print(f"Query indices: {query_idx.tolist()}")
        
        # Visualize the distribution of classes in the support and query sets
        support_labels = labels[support_idx]
        query_labels = labels[query_idx]
        
        support_counter = Counter(support_labels)
        query_counter = Counter(query_labels)
        
        print("Support labels:", support_labels)
        print("Query labels:", query_labels)
        
        fig, axes = plt.subplots(1, 2, figsize=(10, 3))
        x, h = zip(*Counter(support_labels.tolist()).items())
        axes[0].bar(x, h)
        axes[0].set_title('Support Set Class Distribution')
        
        x, h = zip(*Counter(query_labels.tolist()).items())
        axes[1].bar(x, h)
        axes[1].set_title('Query Set Class Distribution')
        # plt.savefig(f"episode_{episode_idx}_distribution.png")
        plt.show()
########################################################
############  End of implementation check ##############
########################################################





