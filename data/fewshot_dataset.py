from torch.utils.data import Dataset

class FewShotDataset(Dataset):
    def __init__(self, root, dataset_name, split, transform=None, split_file=None):
        """
        Few-Shot Dataset Class
        Args:
            root (str): Root path like './database'
            dataset_name (str): Name of the dataset
            split (str): One of ['train', 'val', 'test']
            transform (callable): Transform function from get_transform()
            split_file (str): Optional split txt file (class list)
        """
        self.root = root
        self.dataset_name = dataset_name
        self.split = split
        self.transform = transform
        self.split_file = split_file

        self.data = []
        self.label2idx = {}
        self.idx2label = {}
        self.class_to_images = {}

        self._load_dataset()

    def _load_dataset(self):
        import os
        if self.split_file:
            with open(self.split_file, 'r') as f:
                class_list = [line.strip() for line in f.readlines()]
        else:
            split_path = os.path.join(self.root, self.dataset_name, self.split)
            class_list = [d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))]
        
        class_list.sort()
        for label_idx, class_name in enumerate(class_list):
            self.label2idx[class_name] = label_idx
            self.idx2label[label_idx] = class_name
            class_path = os.path.join(self.root, self.dataset_name, self.split, class_name)

            if not os.path.exists(class_path):
                continue

            image_paths = [
                os.path.join(class_path, fname)
                for fname in os.listdir(class_path)
                if fname.endswith(('.png', '.jpg', '.jpeg'))
            ]

            for img_path in image_paths:
                self.data.append((img_path, label_idx))

            self.class_to_images[label_idx] = image_paths

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        from PIL import Image
        img_path, label = self.data[index]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label
