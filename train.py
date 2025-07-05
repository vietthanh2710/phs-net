import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from torch.utils.data import DataLoader
from torchvision import transforms

import numpy as np
import yaml
import logging

from Utils.dataset import SkinDataset
from Model.model import PHSNet
from Utils.metrics import iou_score, dice_score
from Utils.losses import DiceLoss, TverskyLoss, FocalTverskyLoss, BCETverskyLoss, DiceTverskyLoss

from sklearn.model_selection import train_test_split

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
logs = set()

def init_log(name, level=logging.INFO):
    if (name, level) in logs:
        return
    logs.add((name, level))
    logger = logging.getLogger(name)
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    if "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        logger.addFilter(lambda record: rank == 0)
    else:
        rank = 0
    format_str = "[%(asctime)s][%(levelname)8s] %(message)s"
    formatter = logging.Formatter(format_str)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, length=0):
        self.length = length
        self.reset()

    def reset(self):
        if self.length > 0:
            self.history = []
        else:
            self.count = 0
            self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0

    def update(self, val, num=1):
        if self.length > 0:
            assert num == 1
            self.history.append(val)
            if len(self.history) > self.length:
                del self.history[0]
            self.val = self.history[-1]
            self.avg = np.mean(self.history)
        else:
            self.val = val
            self.sum += val * num
            self.count += num
            self.avg = self.sum / self.count

def evaluate(model, loader, device=device):
    model.eval()
    dice_meter = AverageMeter()
    iou_meter = AverageMeter()
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            images, masks, _ = batch
            images = images.to(device)
            masks = masks.to(device)
            preds = model(images)
            dice = dice_score(preds, masks)
            iou = iou_score(preds, masks)
            dice_meter.update(dice)
            iou_meter.update(iou)
    return dice_meter.avg, iou_meter.avg

# Load your data (modify to match your data)
data = np.load("path/to/data.npy")
images, masks = data["image"], data["mask"]

x_train, x_val, y_train, y_val = train_test_split(images, masks, test_size=0.2, random_state=42)

train_dataset = SkinDataset(x_train, y_train, transform=True, typeData="train")
val_dataset = SkinDataset(x_val, y_val, transform=False, typeData="val")

train_loader = DataLoader(
    train_dataset,
    batch_size=config["batch_size"],
    shuffle=True,
    num_workers=4
)
val_loader = DataLoader(
    val_dataset,
    batch_size=config["batch_size"],
    shuffle=False,
    num_workers=4
)

# Load the model
model = PHSNet(in_channels=3, num_classes=1)
model.to(device)

loss_dict = {
    "DiceLoss": DiceLoss(smooth=1e-5),
    "TverskyLoss": TverskyLoss(alpha=config.get("alpha", 0.7), smooth=1e-5),
    "FocalTverskyLoss": FocalTverskyLoss(alpha=0.7, gamma= 0.75, smooth=1e-5),
    "BCETverskyLoss": BCETverskyLoss(bce_weight=0.5, alpha=0.7, smooth=1e-5),
    "DiceTverskyLoss": DiceTverskyLoss(dice_weight=0.5, alpha=0.7, smooth=1e-5),
}
criterion = loss_dict[config["loss_type"]]

allowed_losses = {"DiceLoss", "TverskyLoss", "FocalTverskyLoss", "BCETverskyLoss", "DiceTverskyLoss"}
assert config["loss_type"] in allowed_losses, f"Invalid loss_type: {config['loss_type']}. Must be one of {allowed_losses}"

criterion = loss_dict[config["loss_type"]]
optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

if __name__ == "__main__":
    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    save_dir = config.get("save_dir", "")
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "best_model.pth") if save_dir else "best_model.pth"

    iters = 0
    total_iters = len(trainloader) * config['epochs']
    best_val_dice = 0.0
    best_epoch = 0
    epoch = -1

    for epoch in range(config["epochs"]):
        logger.info('===========> Epoch: {:}, LR: {:.5f}, Previous best: {:.2f} at epoch {:}'.format(
                epoch, lr, best_val_dice, best_epoch))

        model.train()
        total_loss = AverageMeter()
        
        for i, (image, mask, _) in enumerate(trainloader):
            image, mask = image.to(device), mask.to(device)

            pred = model(image)
            loss = criterion(pred, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss.update(loss)
            iters = epoch * len(trainloader) + i
            lr = config['lr'] * (1 - iters / total_iters) ** 0.9

            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            if (i % (max(2, len(trainloader) // 8)) == 0):
                logger.info('Iters: {:}, Total loss: {:.3f}'.format(i, total_loss.avg))

        val_dice, val_iou = evaluate(model, val_loader, device)
        logger.info(f"Epoch {epoch+1}/{config['epochs']} | Train Loss: {train_loss:.4f} | Val Dice: {val_dice:.4f} | Val IoU: {val_iou:.4f}")

        if val_dice >= best_val_dice:
            best_val_dice = val_dice
            best_epoch = epoch
            torch.save(model.state_dict(), save_path)
    print("Training complete.")