# evaluate_model.py
import torch
from torch.utils.data import DataLoader
from clean_transform_data import ARCDataset
from model import ImprovedCNN


def evaluate_model(model, data_loader):
    """Evaluate the model on validation data."""
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            outputs = model(inputs)
            predicted = torch.round(outputs)
            total_correct += torch.sum(predicted == targets).item()
            total_samples += targets.numel()
    accuracy = total_correct / total_samples
    print(f"Accuracy: {accuracy * 100:.2f}%")
    return accuracy


if __name__ == "__main__":
    from load_data import load_datasets

    _, _, evaluation_challenges, evaluation_solutions, _ = load_datasets()
    validation_dataset = ARCDataset(evaluation_challenges, evaluation_solutions)
    validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False)

    model = ImprovedCNN()
    model.load_state_dict(torch.load('trained_model.pth'))  # Load trained model
    evaluate_model(model, validation_loader)
