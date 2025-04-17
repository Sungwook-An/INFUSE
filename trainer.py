import torch
from tqdm import tqdm
from utils.metrics import AverageMeter, accuracy

def train_one_epoch(args, model, dataloader, optimizer, criterion, device):
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for episode_data in tqdm(dataloader, desc="Training"):
        data, label = episode_data
        
        n_way = args.n_way
        k_shot = args.k_shot
        q_query = args.q_query
        
        total_per_class = k_shot + q_query
        support_indices = []
        query_indices = []
        
        for i in range(n_way):
            start_idx = i * total_per_class
            end_idx = start_idx + k_shot
            support_indices.extend(range(start_idx, end_idx))
            query_indices.extend(range(end_idx, start_idx + total_per_class))
        
        # Create support and query sets
        support_images = data[support_indices]
        support_labels = label[support_indices]
        query_images = data[query_indices]
        query_labels = label[query_indices]
        
        # Create class names
        unique_classes = torch.unique(support_labels).tolist()
        class_names = [dataloader.dataset.idx2label[c] for c in unique_classes]

        # Move to device
        support_images, support_labels = support_images.to(device), support_labels.to(device)
        query_images, query_labels = query_images.to(device), query_labels.to(device)

        # Forward
        logits = model(
            support_images=support_images,
            support_labels=support_labels,
            query_images=query_images,
            class_names=class_names
        )
        
        loss = criterion(logits, query_labels)
        acc = accuracy(logits, query_labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item())
        acc_meter.update(acc.item())

    return loss_meter.avg, acc_meter.avg

def evaluate(args, model, dataloader, criterion, device):
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    with torch.no_grad():
        for episode_data in tqdm(dataloader, desc="Evaluating"):
            support_images, support_labels, query_images, query_labels, class_names = episode_data

            support_images, support_labels = support_images.to(device), support_labels.to(device)
            query_images, query_labels = query_images.to(device), query_labels.to(device)

            logits = model(
                support_images=support_images,
                support_labels=support_labels,
                query_images=query_images,
                class_names=class_names
            )

            loss = criterion(logits, query_labels)
            acc = accuracy(logits, query_labels)

            loss_meter.update(loss.item())
            acc_meter.update(acc.item())

    return loss_meter.avg, acc_meter.avg