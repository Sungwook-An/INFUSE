from data.fewshot_dataset import FewShotDataset

class FC100(FewShotDataset):
    def __init__(self, root, split='train', transform=None):
        """
        FC100 Dataset Class
        
        Args:
            root (str): Root directory of the dataset.
            split (str): One of ['train', 'val', 'test'].
            transform (callable, optional): Image transform.
        """
        dataset_name = 'FC100'
        
        super().__init__(
            root=root,
            dataset_name=dataset_name,
            split=split,
            transform=transform
        )