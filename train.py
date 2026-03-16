import argparse
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch import amp
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import PetACDataset
from src.losses import sprint1_loss
from src.model_unet3d import UNet3D
from src.utils import ensure_dirs, load_config, set_seed


def run_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device,
    train=True,
    use_amp=True,
    grad_clip=1.0,
    loss_cfg=None,
):
    model.train(train)
    losses = []
    loss_cfg = loss_cfg or {}

    for x, y, _, _, _ in tqdm(loader, disable=False):
        x = x.to(device)
        y = y.to(device)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with amp.autocast(device_type=device.type, enabled=use_amp and device.type == "cuda"):
            pred = model(x)
            loss = sprint1_loss(
                pred,
                y,
                w_l1=loss_cfg.get("w_l1", 0.55),
                w_grad=loss_cfg.get("w_grad", 0.15),
                w_hi=loss_cfg.get("w_hi", 0.15),
                w_topk=loss_cfg.get("w_topk", 0.10),
                w_rel=loss_cfg.get("w_rel", 0.10),
                w_log=loss_cfg.get("w_log", 0.10),
                w_peak=loss_cfg.get("w_peak", 0.10),
                w_under=loss_cfg.get("w_under", 0.00),
                hi_threshold=loss_cfg.get("hi_threshold", 0.8),
                hi_boost=loss_cfg.get("hi_boost", 3.0),
                roi_quantile=loss_cfg.get("roi_quantile", 0.99),
                roi_min_voxels=loss_cfg.get("roi_min_voxels", 256),
                peak_voxels=loss_cfg.get("peak_voxels", 64),
                topk_frac=loss_cfg.get("topk_frac", 0.01),
                topk_min_voxels=loss_cfg.get("topk_min_voxels", 256),
                rel_eps=loss_cfg.get("rel_eps", 1e-3),
                log_alpha=loss_cfg.get("log_alpha", 20.0),
                base_clip_max=loss_cfg.get("base_clip_max", 1.0),
            )

        if train:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

        losses.append(loss.item())

    return float(sum(losses) / max(len(losses), 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    norm_cfg = cfg.get("normalization", {})
    input_clip_max = norm_cfg.get("input_clip_max", 1.0)
    target_clip_max = norm_cfg.get("target_clip_max", 1.0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = PetACDataset(
        Path(cfg["data"]["root"]) / "train",
        augment=True,
        input_clip_max=input_clip_max,
        target_clip_max=target_clip_max,
    )
    val_ds = PetACDataset(
        Path(cfg["data"]["root"]) / "val",
        augment=False,
        input_clip_max=input_clip_max,
        target_clip_max=target_clip_max,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=device.type == "cuda",
    )

    model = UNet3D(
        in_channels=cfg["model"]["in_channels"],
        out_channels=cfg["model"]["out_channels"],
        base=cfg["model"]["base_channels"],
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    scheduler_cfg = cfg.get("scheduler", {})
    scheduler = None
    if scheduler_cfg.get("enabled", True):
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=scheduler_cfg.get("factor", 0.5),
            patience=scheduler_cfg.get("patience", 3),
            min_lr=scheduler_cfg.get("min_lr", 1e-6),
        )
    scaler = amp.GradScaler(device=device.type, enabled=cfg["train"]["amp"] and device.type == "cuda")

    resume_ckpt = args.resume or cfg["train"].get("resume_ckpt")
    if resume_ckpt:
        resume_path = Path(resume_ckpt)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        state = torch.load(resume_path, map_location=device)
        try:
            model.load_state_dict(state["model_state_dict"], strict=True)
        except RuntimeError as e:
            raise RuntimeError(
                "Failed to load resume checkpoint due to architecture mismatch. "
                "Use a config with the same model shape (e.g., base_channels) "
                f"as the checkpoint: {resume_path}"
            ) from e
        print(f"Resumed model weights from {resume_path}")
        if cfg["train"].get("resume_optimizer", False):
            if "optimizer_state_dict" in state:
                optimizer.load_state_dict(state["optimizer_state_dict"])
            if "scaler_state_dict" in state and scaler.is_enabled():
                scaler.load_state_dict(state["scaler_state_dict"])
            print("Resumed optimizer/scaler state.")

    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    ensure_dirs(ckpt_dir)

    best_val = float("inf")
    best_epoch = 0
    no_improve_epochs = 0
    es_patience = cfg["train"].get("early_stop_patience", 8)
    es_min_delta = cfg["train"].get("early_stop_min_delta", 1e-4)
    history = []

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        print(f"Epoch {epoch}/{cfg['train']['epochs']}")
        train_loss = run_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            train=True,
            use_amp=cfg["train"]["amp"],
            grad_clip=cfg["train"]["grad_clip"],
            loss_cfg=cfg.get("loss", {}),
        )
        with torch.no_grad():
            val_loss = run_epoch(
                model,
                val_loader,
                optimizer,
                scaler,
                device,
                train=False,
                use_amp=cfg["train"]["amp"],
                loss_cfg=cfg.get("loss", {}),
            )

        if scheduler is not None:
            scheduler.step(val_loss)
        current_lr = float(optimizer.param_groups[0]["lr"])

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": current_lr})
        print(f"train_loss={train_loss:.5f} val_loss={val_loss:.5f} lr={current_lr:.7f}")

        if val_loss < best_val - es_min_delta:
            best_val = val_loss
            best_epoch = epoch
            no_improve_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict() if scaler.is_enabled() else None,
                    "config": cfg,
                    "epoch": epoch,
                    "best_val": best_val,
                },
                ckpt_dir / "best_model.pt",
            )
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= es_patience:
                print(
                    f"Early stopping at epoch {epoch}. "
                    f"Best epoch={best_epoch} best_val={best_val:.5f}"
                )
                break

    pd.DataFrame(history).to_csv(ckpt_dir / "train_log.csv", index=False)
    print("Training complete. Saved checkpoints/best_model.pt")


if __name__ == "__main__":
    main()
