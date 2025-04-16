# main.py

import torch
from torch import nn, optim
from data.dataset import get_dataset
from models.infuse_model import INFUSEModel
from trainer import train_one_epoch, evaluate
from torch.utils.data import DataLoader
from utils.episode import FewShotEpisodeSampler
from utils.logger import log_metrics

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = get_dataset(args.dataset, split='train', transform=args.train_transform, root=args.data_root)
    val_dataset = get_dataset(args.dataset, split='val', transform=args.test_transform, root=args.data_root)

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=FewShotEpisodeSampler(train_dataset, args.num_episodes, args.n_way, args.k_shot, args.q_query),
        num_workers=4
    )

    val_loader = DataLoader(
        val_dataset,
        batch_sampler=FewShotEpisodeSampler(val_dataset, args.num_val_episodes, args.n_way, args.k_shot, args.q_query),
        num_workers=4
    )

    model = INFUSEModel(args).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        print(f"\n[Epoch {epoch}]")
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.2f}%")

        log_metrics(epoch=epoch,
                    train_loss=train_loss, train_acc=train_acc,
                    val_loss=val_loss, val_acc=val_acc)

        torch.save(model.state_dict(), f"checkpoints/infuse_epoch{epoch}.pth")

if __name__ == "__main__":
    import argparse
    import yaml
    from pathlib import Path

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="fc100")
    parser.add_argument("--data_root", type=str, default="/root/INFUSE/database")
    parser.add_argument("--train_transform", default=None)
    parser.add_argument("--test_transform", default=None)

    parser.add_argument("--n_way", type=int, default=5)
    parser.add_argument("--k_shot", type=int, default=1)
    parser.add_argument("--q_query", type=int, default=15)
    parser.add_argument("--num_episodes", type=int, default=100)
    parser.add_argument("--num_val_episodes", type=int, default=100)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--feature_dim", type=int, default=640)  # Should match EVG output_dim
    parser.add_argument("--metric", type=str, default="cosine", choices=["cosine", "euclidean"])
    parser.add_argument("--classifier_temperature", type=float, default=1.0)

    args = parser.parse_args()

    # Load configs from YAML files
    with open(Path(__file__).parent / "configs/image_encoder.yaml") as f:
        args.image_encoder_config = yaml.safe_load(f)
    
    with open(Path(__file__).parent / "configs/text_encoder.yaml") as f:
        args.text_encoder_config = yaml.safe_load(f)

    with open(Path(__file__).parent / "configs/evg.yaml") as f:
        args.evg_config = yaml.safe_load(f)

    main(args)
