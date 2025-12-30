import os, zipfile, json, random, shutil
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import gradio as gr
from PIL import Image

DATA_DIR = Path("data")
MODEL_DIR = Path("modeller")
MODEL_PATH = MODEL_DIR / "model.pth"
CLASS_PATH = MODEL_DIR / "class_names.json"

ZIP_NAME = "intel_dataset.zip"  # BUNU SONRA YÜKLEYECEĞİZ

BATCH_SIZE = 16
EPOCHS = 2
LR = 1e-4


def prepare_dataset():
    if (DATA_DIR / "split").exists():
        return "Dataset zaten hazır."

    if not Path(ZIP_NAME).exists():
        return "❌ intel_dataset.zip Space'e yüklenmedi."

    extract_dir = DATA_DIR / "unzipped"
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(ZIP_NAME, "r") as z:
        z.extractall(extract_dir)

    seg_train = None
    for p in extract_dir.rglob("seg_train"):
        if p.is_dir():
            seg_train = p
            break

    if seg_train is None:
        return "❌ seg_train klasörü bulunamadı."

    raw = DATA_DIR / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    for cls in seg_train.iterdir():
        if cls.is_dir():
            dst = raw / cls.name
            dst.mkdir(parents=True, exist_ok=True)
            for img in cls.iterdir():
                if img.is_file():
                    shutil.copy2(img, dst / img.name)

    out = DATA_DIR / "split"
    for sp in ["train", "val", "test"]:
        (out / sp).mkdir(parents=True, exist_ok=True)

    IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

    for cls in raw.iterdir():
        imgs = [p for p in cls.iterdir() if p.suffix.lower() in IMG_EXTS]
        random.shuffle(imgs)
        n = len(imgs)
        tr = imgs[:int(n*0.8)]
        va = imgs[int(n*0.8):int(n*0.9)]
        te = imgs[int(n*0.9):]

        for p in tr:
            shutil.copy2(p, out/"train"/cls.name/p.name)
        for p in va:
            shutil.copy2(p, out/"val"/cls.name/p.name)
        for p in te:
            shutil.copy2(p, out/"test"/cls.name/p.name)

    return "✅ Dataset hazırlandı."


def train_model():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if MODEL_PATH.exists():
        return "Model zaten var."

    msg = prepare_dataset()
    if msg.startswith("❌"):
        return msg

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tf_train = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    tf_val = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    train_ds = datasets.ImageFolder(str(DATA_DIR/"split/train"), tf_train)
    val_ds = datasets.ImageFolder(str(DATA_DIR/"split/val"), tf_val)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(train_ds.classes))
    model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=LR)

    best = 0.0
    for _ in range(EPOCHS):
        model.train()
        for x,y in train_dl:
            x,y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = loss_fn(out,y)
            loss.backward()
            opt.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x,y in val_dl:
                x,y = x.to(device), y.to(device)
                pred = model(x).argmax(1)
                correct += (pred==y).sum().item()
                total += y.size(0)

        acc = correct/total
        if acc > best:
            best = acc
            torch.save(model.state_dict(), MODEL_PATH)
            with open(CLASS_PATH,"w") as f:
                json.dump(train_ds.classes,f)

    return f"✅ Eğitim tamamlandı (acc={best:.2f})"


def predict(img):
    if img is None:
        return "Görsel yükle", {}

    if not MODEL_PATH.exists():
        train_model()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(CLASS_PATH) as f:
        classes = json.load(f)

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)

    x = tf(img).unsqueeze(0).to(device)
    probs = F.softmax(model(x), dim=1)[0].cpu().tolist()

    idx = probs.index(max(probs))
    return classes[idx], {classes[i]: probs[i] for i in range(len(classes))}


gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=[gr.Textbox(label="Tahmin"), gr.Label(label="Olasılıklar")],
    title="Scene Classification Demo",
    description="Hazır dataset + ResNet18 + Gradio"
).launch()
