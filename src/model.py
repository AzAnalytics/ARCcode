# model.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from clean_transform_data import ARCDataset


class ImprovedCNN(nn.Module):
    """Improved CNN model for ARC challenges."""

    def __init__(self):
        super(ImprovedCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 30 * 30, 256)
        self.fc2 = nn.Linear(256, 900)  # Output for 30x30 grid

    def forward(self, x):
        x = torch.relu(self.conv1(x.unsqueeze(1)))  # Add channel dimension
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.view(-1, 30, 30)
        return x


def train_model(model, train_loader, num_epochs=20, learning_rate=0.001):
    """Train the CNN model."""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        total_loss = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader)}")


if __name__ == "__main__":
    from load_data import load_datasets

    training_challenges, training_solutions, _, _, _ = load_datasets()
    train_dataset = ARCDataset(training_challenges, training_solutions)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    model = ImprovedCNN()
    train_model(model, train_loader)
