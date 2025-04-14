# main.py

import torch
from torch import nn, optim
from data.dataset import get_dataset
from models.infuse_model import INFUSEModel  # 당신의 전체 모델
from trainer import train_one_epoch, evaluate
from torch.utils.data import DataLoader
from utils.episode import FewShotEpisodeSampler  # 에피소드 샘플링 함수
from utils.logger import log_metrics  # wandb나 텐서보드 대체 가능

def main(args):
    # 환경 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 데이터셋 로드
    train_dataset = get_dataset(args.dataset, split='train', transform=args.train_transform, root=args.data_root)
    val_dataset = get_dataset(args.dataset, split='val', transform=args.test_transform, root=args.data_root)

    # 에피소드 샘플링 기반 DataLoader
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

    # 모델 초기화
    model = INFUSEModel(args).to(device)

    # 손실함수 및 옵티마이저
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 학습 루프
    for epoch in range(1, args.epochs + 1):
        print(f"\n[Epoch {epoch}]")
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.2f}%")

        # wandb 또는 tensorboard 사용 시
        log_metrics(epoch=epoch,
                    train_loss=train_loss, train_acc=train_acc,
                    val_loss=val_loss, val_acc=val_acc)

        # 모델 저장 (선택)
        torch.save(model.state_dict(), f"checkpoints/infuse_epoch{epoch}.pth")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    # 데이터 설정
    parser.add_argument("--dataset", type=str, default="fc100")
    parser.add_argument("--data_root", type=str, default="~/data/FC100")
    parser.add_argument("--train_transform", default=None)
    parser.add_argument("--test_transform", default=None)

    # Few-shot 설정
    parser.add_argument("--n_way", type=int, default=5)
    parser.add_argument("--k_shot", type=int, default=1)
    parser.add_argument("--q_query", type=int, default=15)
    parser.add_argument("--num_episodes", type=int, default=100)
    parser.add_argument("--num_val_episodes", type=int, default=100)

    # 학습 설정
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)

    args = parser.parse_args()
    main(args)
