# trainer.py

import torch
from tqdm import tqdm
from utils.metrics import AverageMeter, accuracy

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for episode_data in tqdm(dataloader, desc="Training"):
        support_images, support_labels, query_images, query_labels, class_names = episode_data

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

def evaluate(model, dataloader, criterion, device):
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