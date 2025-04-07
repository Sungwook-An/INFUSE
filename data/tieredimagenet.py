from data.fewshot_dataset import FewShotDataset

class TieredImageNet(FewShotDataset):
    def __init__(self, root, split='train', transform=None, split_file=None):
        """
        TieredImageNet Dataset Class

        Args:
            root (str): Root path like './database'
            split (str): One of ['train', 'val', 'test']
            transform (callable): Transform function from get_transform()
            split_file (str): Optional split txt file (class list)
        """
        dataset_name = 'TieredImageNet'

        super().__init__(
            root=root,
            dataset_name=dataset_name,
            split=split,
            transform=transform,
            split_file=split_file
        )
