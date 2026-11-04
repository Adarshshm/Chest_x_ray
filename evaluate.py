import os
import json
import torch
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from dataset import get_dataloaders
from model import TBXNet

def predict_single_image(image_path, model_path="best_tbxnet_model.pth", base_dir="./dataset"):
    """
    Step-by-step processor for making clinical predictions on a single chest X-ray image:
    
    Step 1: Dynamically loads the directory class labels.
    Step 2: Detects and selects the available hardware computing device (GPU or CPU).
    Step 3: Opens the target image file and converts it to a standard RGB format.
    Step 4: Resizes the image dimensions to 128x128 pixels to match model input requirements.
    Step 5: Converts the image into a floating-point numpy array and scales pixel values to [0, 1].
    Step 6: Permutes channel layout and expands dimensions to form a PyTorch tensor of shape (1, C, H, W).
    Step 7: Instantiates the TBXNet model architecture and loads trained state weights from disk.
    Step 8: Sets the model to evaluation mode and performs forward pass inference.
    Step 9: Applies SoftMax activation to compute normalized class prediction probabilities and confidence.
    """
    _, _, _, num_classes, class_names = get_dataloaders(base_dir, batch_size=32)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    img = Image.open(image_path).convert('RGB')
    img_resized = img.resize((128, 128))
    img_arr = np.array(img_resized, dtype=np.float32) / 255.0
    
    img_tensor = torch.tensor(img_arr, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
    
    model = TBXNet(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, dim=1)
        
    predicted_class = class_names[predicted_idx.item()]
    confidence_score = confidence.item() * 100
    
    return predicted_class, confidence_score

def evaluate_model():
    """
    Evaluates model performance across test data splits, computes evaluation metrics,
    plots and saves a confusion matrix, and generates a graphical representation
    of training and validation loss/accuracy curves.
    """
    data_dir = "./dataset"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_loader, num_classes, class_names = get_dataloaders(data_dir, batch_size=32)

    model = TBXNet(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load("best_tbxnet_model.pth", map_location=device))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    cm = confusion_matrix(y_true, y_pred)
    specificity_list = []
    for i in range(len(cm)):
        true_negatives = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]
        false_positives = np.sum(cm[:, i]) - cm[i, i]
        specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0
        specificity_list.append(specificity)
    avg_specificity = np.mean(specificity_list)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title("TBXNet Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig("confusion_matrix.png")
    plt.close()

    if os.path.exists("training_history.json"):
        with open("training_history.json", "r") as f:
            history = json.load(f)

        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Loss Graphical Representation')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history['train_acc'], label='Training Accuracy')
        plt.plot(history['val_acc'], label='Validation Accuracy')
        plt.title('Accuracy Graphical Representation')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig("training_performance_curves.png")
        plt.close()

if __name__ == "__main__":
    evaluate_model()