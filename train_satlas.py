import torch
from torch.utils.data import DataLoader

from PIL import Image
import os

import satlaspretrain_models

import warnings
warnings.filterwarnings('ignore')

from torchvision import transforms
from tqdm import tqdm

import kagglehub
import json

class WildfireDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, name:str = ""):
        self.data = []
        self.name = name
        self.root_dir = root_dir
        self.transform = transforms.Compose([
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            # transforms.Normalize(mean=[0, 0, 0], std=np.sqrt(np.array([255, 255, 255]))),
            transforms.Resize((512, 512)),]
        )
        self.classes = os.listdir(root_dir)
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir, cls_name)
            for img_name in tqdm(os.listdir(cls_dir), desc=f"{name}: Loading for class {cls_name}"):
                img_path = os.path.join(cls_dir, img_name)
                img = Image.open(img_path)
                self.data.append({'img': img, 'cls': self.class_to_idx[cls_name]})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]['img']
        label = self.data[idx]['cls']
        img = self.transform(img)
        return {'img': img, 'cls': label}

    def get_data(self, idx):
        return self.data[idx]

path = kagglehub.dataset_download("abdelghaniaaba/wildfire-prediction-dataset")

val_dir = os.path.join(path, "valid")
dataset = WildfireDataset(val_dir, "val")

with open('indices.json') as f:
    indices = json.load(f)

train_indices = indices['train']
val_indices = indices['val']

train_dataset = torch.utils.data.Subset(dataset, train_indices)
val_dataset = torch.utils.data.Subset(dataset, val_indices)

# train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)

import torch
import os
import satlaspretrain_models
import warnings
warnings.filterwarnings('ignore')


def load_model(
        model_identifier="Sentinel2_SwinB_SI_RGB",
        fpn=True,
        train_backbone=True,
        train_fpn=True,
        load_model_path=None,
        lr=1e-5,
        weight_decay=1e-5,
        device="cpu"):
    
    if not os.path.isdir("trained_models"):
        os.mkdir("trained_models")
    

    weights_manager = satlaspretrain_models.Weights()
    model = weights_manager.get_pretrained_model(model_identifier=model_identifier, fpn=fpn, head=satlaspretrain_models.utils.Head.CLASSIFY,num_categories=2)
    model.to(device)

    if load_model_path is not None:
        if not "trained_models/" in load_model_path:
            load_model_path = "trained_models/" + load_model_path
        if os.path.exists(load_model_path):
            model.load_state_dict(torch.load(load_model_path, map_location=device))
    
    for n, p in model.named_parameters():
        if 'backbone' in n:
            p.requires_grad = train_backbone
        if fpn:
            if 'fpn' in n:
                p.requires_grad = train_fpn
        # print(n, p.requires_grad)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    return model, optimizer


def train(model, optimizer, num_epochs, model_name="model"):
    model.train()

    device = next(model.parameters()).device

    best_accuracy = 0

    for epoch in range(num_epochs):
        print(f"Epoch: {epoch}")
        total_loss = 0
        pb = tqdm(total=len(train_loader), desc="Training")
        model.train()
        for batch, d in enumerate(train_loader):
            X = d['img'].to(device)
            y = d['cls'].to(device)
            _, loss = model(X, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pb.update(1)
            pb.set_postfix({"Loss": total_loss / (batch + 1)})
        print(f"Train Loss: {total_loss / len(train_loader)}")

        model.eval()
        pb = tqdm(total=len(val_loader), desc="Validation")
        tp, tn, fp, fn = 0, 0, 0, 0
        total_loss = 0
        for batch, d in enumerate(val_loader):
            X = d['img'].to(device)
            y = d['cls'].to(device)
            with torch.no_grad():
                y_pred, loss = model(X, y)
            total_loss += loss.item()
            pred = (y_pred[:, 1] > y_pred[:, 0]).int()
            tp += ((y == 1) & (pred == 1)).sum().item()
            tn += ((y == 0) & (pred == 0)).sum().item()
            fp += ((y == 0) & (pred == 1)).sum().item()
            fn += ((y == 1) & (pred == 0)).sum().item()
            F1_score = 2 * tp / (2 * tp + fp + fn)
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            pb.update(1)
            pb.set_postfix({"Accuracy": accuracy, "F1-score": F1_score, "Loss": total_loss / (batch+1)})
        print(f"Accuracy: {accuracy}, F1-score: {F1_score}, Loss: {total_loss / len(val_loader)}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), "trained_models/" + model_name + "_best.pth")

    torch.save(model.state_dict(), "trained_models/" + model_name + "_last.pth")
        
if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()

    parser.add_argument("--model_id", type=str, default="Sentinel2_SwinB_SI_RGB", help="Model identifier")
    parser.add_argument("--fpn", action="store_true", help="Use FPN")
    parser.add_argument("--train_backbone", action="store_true", help="Train backbone")
    parser.add_argument("--train_fpn", action="store_true", help="Train FPN")
    parser.add_argument("--load_model", type=str, default=None, help="Load model path")

    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="L2 weight decay")
    parser.add_argument("--model_name", type=str, default="model", help="Model name")

    args = parser.parse_args()

    model, optimizer = load_model(
        model_identifier=args.model_id,
        fpn=args.fpn,
        train_backbone=args.train_backbone,
        train_fpn=args.train_fpn,
        load_model_path=args.load_model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    train(model, optimizer, args.num_epochs, args.model_name)

