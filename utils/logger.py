def log_metrics(epoch, train_loss, train_acc, val_loss, val_acc):
    try:
        import wandb
        wandb.log({
            "epoch": epoch,
            "train/loss": train_loss,
            "train/acc": train_acc,
            "val/loss": val_loss,
            "val/acc": val_acc
        })
    except ImportError:
        print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}")
