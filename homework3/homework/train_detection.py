import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils.tensorboard as tb

from .models import ClassificationLoss, load_model, save_model, RegressionLoss
from .datasets.road_dataset import load_data


def train(
        exp_dir: str = "logs",
        model_name: str = "detector",
        num_epoch: int = 200,
        lr: float = 1e-1,
        batch_size: int = 128,
        seed: int = 2024,
        num_layers: int = 20,
        **kwargs,
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    # set random seed so each run is deterministic
    torch.manual_seed(seed)
    np.random.seed(seed)

    # directory with timestamp to save tensorboard logs and model checkpoints
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    # note: the grader uses default kwargs, you'll have to bake them in for the final submission
    model = load_model(model_name, **kwargs)
    model = model.to(device)
    model.train()

    train_data = load_data("datasets/road_data/train", shuffle=True, batch_size=batch_size, num_workers=4)
    val_data = load_data("datasets/road_data/val", shuffle=False, batch_size=batch_size, num_workers=4)

    # create loss function and optimizer
    loss_func_seg = ClassificationLoss()
    loss_func_depth = RegressionLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # optimizer = ...
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum =0 )

    global_step = 0
    metrics = {"train_acc": [], "val_acc": []}

    # training loop
    for epoch in range(num_epoch):
        # clear metrics at beginning of epoch
        for key in metrics:
            metrics[key].clear()
        model.train()

        for batch in train_data:
            img = batch['image'].to(device)
            depth = batch['depth'].to(device)
            track = batch['track'].to(device)  # track represents the segmentation labels

            # TODO: implement training step

            # Forward pass
            seg_out, depth_out = model(img)

            # Compute segmentation loss
            loss_seg = loss_func_seg(seg_out, track)

            # Compute depth regression loss
            loss_depth = loss_func_depth(depth_out, depth)

            total_loss = loss_seg + loss_depth

            # Backpropagation
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # calculate train accuracy + store to metrics

            pred = torch.argmax(seg_out, dim=1)
            correct = (pred == track).sum().item()
            accuracy = correct / track.numel()
            metrics["train_acc"].append(accuracy)

            logger.add_scalar('Loss/segmentation', loss_seg.item(), global_step)
            logger.add_scalar('Loss/depth', loss_depth.item(), global_step)
            logger.add_scalar('Accuracy/train', accuracy, global_step)

            global_step += 1

        # disable gradient computation and switch to evaluation mode
        with torch.inference_mode():
            model.eval()
            total_correct = 0
            total_samples = 0

            for batch in val_data:
                img = batch['image'].to(device)
                depth = batch['depth'].to(device)
                track = batch['track'].to(device)  # track represents the segmentation labels

                # TODO: compute validation accuracy
                seg_out, depth_out = model(img)

                # Compute validation accuracy
                pred = torch.argmax(seg_out, dim=1)
                correct = (pred == track).sum().item()
                total_correct += correct
                total_samples += track.numel()

            val_accuracy = total_correct / total_samples
            metrics["val_acc"].append(val_accuracy)

        # log average train and val accuracy to tensorboard
        epoch_train_acc = torch.as_tensor(metrics["train_acc"]).mean()
        epoch_val_acc = torch.as_tensor(metrics["val_acc"]).mean()

        # print on first, last, every 10th epoch
        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
                f"train_acc={epoch_train_acc:.4f} "
                f"val_acc={epoch_val_acc:.4f}"
            )

    # save and overwrite the model in the root directory for grading
    save_model(model)

    # save a copy of model weights in the log directory
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_epoch", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2024)

    # optional: additional model hyperparamters
    parser.add_argument("--num_layers", type=int, default=3)

    # pass all arguments to train
    train(**vars(parser.parse_args()))