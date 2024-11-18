import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau

class MmWaveDataset(Dataset):
    def __init__(self, features, labels, transform=None):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.transform = transform
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.labels[idx]
        
        if self.transform:
            x = self.transform(x)
            
        return x, y

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Linear(in_channels, out_channels)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Linear(out_channels, out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x):
        residual = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out

class AttentionModule(nn.Module):
    def __init__(self, in_features):
        super(AttentionModule, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.ReLU(),
            nn.Linear(in_features // 2, in_features),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        attention_weights = self.attention(x)
        return x * attention_weights
        

class AdvancedMmWaveNN(nn.Module):
    def __init__(self, input_size, num_classes=8):
        super(AdvancedMmWaveNN, self).__init__()
        
        self.hidden_size = 256
        self.num_residual_blocks = 3
        
        # Initial feature extraction
        self.feature_extraction = nn.Sequential(
            nn.Linear(input_size, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Parallel processing branches
        self.spatial_branch = self._create_branch(self.hidden_size // 2)
        self.channel_branch = self._create_branch(self.hidden_size // 2)
        
        # Attention modules
        self.spatial_attention = AttentionModule(self.hidden_size // 2)
        self.channel_attention = AttentionModule(self.hidden_size // 2)
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(self.hidden_size, self.hidden_size)
            for _ in range(self.num_residual_blocks)
        ])
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.BatchNorm1d(self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_size // 2, self.hidden_size // 4),
            nn.BatchNorm1d(self.hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_size // 4, num_classes)
        )
        
    def _create_branch(self, branch_size):
        return nn.Sequential(
            nn.Linear(self.hidden_size, branch_size),
            nn.BatchNorm1d(branch_size),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
    
    def forward(self, x):
        x = self.feature_extraction(x)
        
        spatial_features = self.spatial_branch(x)
        channel_features = self.channel_branch(x)
        
        spatial_features = self.spatial_attention(spatial_features)
        channel_features = self.channel_attention(channel_features)
        
        combined_features = torch.cat([spatial_features, channel_features], dim=1)
        
        for block in self.residual_blocks:
            combined_features = block(combined_features)
        
        output = self.classifier(combined_features)
        return output

def train_epoch(model, train_loader, criterion, optimizer, device, clip_value=1.0):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return running_loss / len(train_loader), 100. * correct / total

def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return running_loss / len(test_loader), 100. * correct / total, all_preds, all_labels

def plot_training_history(train_losses, train_acc, val_losses, val_acc, save_path='training_history_advanced.png'):
    plt.style.use('seaborn')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    ax1.plot(train_losses, label='Training Loss', color='blue', alpha=0.7)
    ax1.plot(val_losses, label='Validation Loss', color='red', alpha=0.7)
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracies
    ax2.plot(train_acc, label='Training Accuracy', color='blue', alpha=0.7)
    ax2.plot(val_acc, label='Validation Accuracy', color='red', alpha=0.7)
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    # Load and preprocess data
    X = np.load('mmwave_features.npy')
    y = np.load('mmwave_labels.npy')

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    dataset = MmWaveDataset(X_scaled, y)
    
    # Split data
    train_size = int(0.6 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Setup device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AdvancedMmWaveNN(input_size=X.shape[1]).to(device)
    
    # Setup training components
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # Training loop
    num_epochs = 100
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    best_val_acc = 0
    
    print(f"Training on {device}")
    for epoch in range(num_epochs):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        # Validate
        val_loss, val_acc, _, _ = evaluate_model(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, 'best_mmwave_model_advanced.pth')
        
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}]')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            print('-' * 50)
    
    # Plot training history
    plot_training_history(train_losses, train_accuracies, val_losses, val_accuracies)
    
    # Load best model and evaluate on test set
    checkpoint = torch.load('best_mmwave_model_advanced.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    test_loss, test_acc, y_pred, y_true = evaluate_model(model, test_loader, criterion, device)
    
    print(f'\nBest Model Test Accuracy: {test_acc:.2f}%')

if __name__ == "__main__":
    main()