import os
import random


def make_split_file(root, dataset_name, save_dir, train_ratio=0.6, val_ratio=0.2, seed=42):
    """
    Automatically generate train/val/test class splits and save them as .txt files.

    Args:
        root (str): Root path of the dataset folder (e.g., './database')
        dataset_name (str): Name of the dataset (e.g., 'MiniImageNet')
        save_dir (str): Directory where split files will be saved
        train_ratio (float): Proportion of classes for the training set
        val_ratio (float): Proportion of classes for the validation set
        seed (int): Random seed for reproducibility
    """

    dataset_path = os.path.join(root, dataset_name)
    class_names = sorted([
        d for d in os.listdir(dataset_path) 
        if os.path.isdir(os.path.join(dataset_path, d))
    ])

    random.seed(seed)
    random.shuffle(class_names)

    total_classes = len(class_names)
    num_train = int(total_classes * train_ratio)
    num_val = int(total_classes * val_ratio)

    train_classes = class_names[:num_train]
    val_classes = class_names[num_train:num_train + num_val]
    test_classes = class_names[num_train + num_val:]

    os.makedirs(save_dir, exist_ok=True)

    for split, classes in zip(['train', 'val', 'test'], [train_classes, val_classes, test_classes]):
        file_path = os.path.join(save_dir, f'{dataset_name.lower()}_{split}.txt')
        with open(file_path, 'w') as f:
            for class_name in classes:
                f.write(class_name + '\n')

    print(f"Split files saved in: {save_dir}")
