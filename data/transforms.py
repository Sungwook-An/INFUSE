from torchvision import transforms
import numpy as np

def get_transform(backbone, dataset, split='train'):
    backbone = backbone.lower()
    dataset = dataset.lower()
    split = split.lower()
    print(backbone, dataset, split)
    
    is_train = split == 'train'
    
    if backbone == 'resnet':
        resize_size = 84
        if dataset in ['fc100', 'cifar-fs']:
            normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            # normalize = transforms.Normalize((0.4725, 0.4533, 0.4100), (0.2770, 0.2680, 0.2840))
            if is_train:
                transforms_list = [
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize
                ]
            else:
                transforms_list = [
                    transforms.ToTensor(),
                    normalize
                ]
        elif dataset in ['miniimagenet', 'tieredimagenet']:
            normalize = transforms.Normalize(
                np.array([x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]]),
                np.array([x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]])
            )
            if is_train:
                transforms_list = [
                    transforms.Resize(resize_size),
                    transforms.CenterCrop(resize_size),
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize
                ]
            else:
                transforms_list = [
                    transforms.Resize(resize_size),
                    transforms.CenterCrop(resize_size),
                    transforms.ToTensor(),
                    normalize
                ]
        else:
            raise ValueError(f"Unsupported dataset '{dataset}' for backbone 'resnet'")
        
        
        
    elif backbone == 'swin':
        resize_size = 224
        if dataset in ['fc100', 'cifar-fs']:
            normalize = transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023))
            if is_train:
                transforms_list = [
                    transforms.Resize((resize_size, resize_size)),
                    transforms.CenterCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                    transforms.ToTensor(),
                    normalize
                ]
            else:
                transforms_list = [
                    transforms.Resize((resize_size, resize_size)),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize
                ]
        elif dataset in ['miniimagenet', 'tieredimagenet']:
            normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            if is_train:
                transforms_list = [
                    transforms.Resize((256, 256)),
                    transforms.CenterCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                    transforms.ToTensor(),
                    normalize
                ]
            else:
                transforms_list = [
                    transforms.Resize((256, 256)),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize
                ]
        else:
            raise ValueError(f"Unsupported dataset '{dataset}' for backbone 'swin'")
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    return transforms.Compose(transforms_list)
    
    