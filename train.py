from tqdm import tqdm
import json
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloaders
from model import TBXNet

def train_model():
    """
    Executes training and validation loops using Categorical Cross-Entropy Loss
    and the Stochastic Gradient Descent with Momentum (SGDM) optimizer.
    Saves the best validation checkpoint and exports metrics history for plotting.
    """
    data_dir = "./dataset"
    batch_size = 32
    epochs = 40
    learning_rate = 0.05
    momentum = 0.9

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader, num_classes, class_names = get_dataloaders(data_dir, batch_size)

    model = TBXNet(num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    best_val_acc = 0.0
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        train_progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]", leave=True)
        for images, labels in train_progress_bar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total_train += labels.size(0)
            correct_train += predicted.eq(labels).sum().item()

        train_loss = running_loss / total_train
        train_acc = correct_train / total_train

        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total_val += labels.size(0)
                correct_val += predicted.eq(labels).sum().item()

        val_loss = val_loss / total_val
        val_acc = correct_val / total_val

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_tbxnet_model.pth")

    with open("training_history.json", "w") as f:
        json.dump(history, f)

if __name__ == "__main__":
    train_model()