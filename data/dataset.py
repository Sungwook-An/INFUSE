from data.fc100 import FC100
from data.cifar_fs import CIFAR_FS
from data.miniimagenet import MiniImageNet
from data.tieredimagenet import TieredImageNet


def get_dataset(name, root, split='train', transform=None, split_file=None):
    """
    Dataset dispatcher function

    Args:
        name (str): Dataset name, one of ['fc100', 'cifar-fs', 'miniimagenet', 'tieredimagenet']
        root (str): Root directory (e.g., './database')
        split (str): One of ['train', 'val', 'test']
        transform (callable): Transform function
        split_file (str or None): Optional class split list

    Returns:
        A dataset instance (subclass of FewShotDataset)
    """

    name = name.lower()

    if name == 'fc100':
        return FC100(root=root, split=split, transform=transform)
    elif name == 'cifar-fs':
        return CIFAR_FS(root=root, split=split, transform=transform)
    elif name == 'miniimagenet':
        return MiniImageNet(root=root, split=split, transform=transform, split_file=split_file)
    elif name == 'tieredimagenet':
        return TieredImageNet(root=root, split=split, transform=transform, split_file=split_file)
    else:
        raise ValueError(f"Unsupported dataset name: {name}")

 

########################################################
#############  Checking the implementation #############
########################################################
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    
    from data.transforms import get_transform
    
    dataset_name = 'fc100'
    root_path = './database'
    split = 'train'
    backbone = 'resnet'
    
    transform = get_transform(backbone=backbone, dataset=dataset_name, split=split)
    dataset = get_dataset(
        name=dataset_name,
        root=root_path,
        split=split,
        transform=transform,
    )
    
    print(f"Dataset: {dataset_name}, Split: {split}")
    print(f"Total samples: {len(dataset)}")
    
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    for images, labels in loader:
        print("Image batch shape:", images.shape)
        print("Label batch      :", labels)
        break
########################################################
############  End of implementation check ##############
########################################################