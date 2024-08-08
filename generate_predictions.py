# generate_predictions.py
import torch
import json
from torch.utils.data import DataLoader
from clean_transform_data import ARCDataset
from model import ImprovedCNN


def generate_predictions(model, data_loader):
    """Generate predictions for test data."""
    model.eval()
    predictions = {}
    with torch.no_grad():
        for inputs, task_ids in data_loader:
            # Assurez-vous que task_ids est récupéré correctement
            task_id = task_ids[0]  # récupérer le task_id à partir de la liste
            outputs = model(inputs)
            predicted = torch.round(outputs).int().tolist()  # Convert predictions to a list of integers
            # Ajouter les tentatives de prédiction à la tâche
            predictions[task_id] = [{"attempt_1": predicted, "attempt_2": predicted}]
    return predictions


def write_predictions_to_json(predictions, file_path):
    """Write predictions to a JSON file with compact formatting."""
    with open(file_path, 'w') as file:
        # Use separators to eliminate spaces after commas and colons
        json.dump(predictions, file, separators=(',', ':'))


if __name__ == "__main__":
    # Charger les datasets de test
    from load_data import load_datasets

    _, _, _, _, test_challenges = load_datasets()

    # Préparer le dataset et le DataLoader
    test_dataset = ARCDataset(test_challenges)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Charger le modèle entraîné
    model = ImprovedCNN()
    model.load_state_dict(torch.load('trained_model.pth'))  # Charger le modèle sauvegardé
    model.eval()  # Mettre le modèle en mode évaluation

    # Générer les prédictions
    predictions = generate_predictions(model, test_loader)

    # Écrire les prédictions dans un fichier JSON avec un format compact
    submission_file_path = 'sample_submission.json'
    write_predictions_to_json(predictions, submission_file_path)
    print(f"Prédictions écrites dans le fichier {submission_file_path}.")

    # Afficher un aperçu du fichier de soumission
    with open(submission_file_path, 'r') as file:
        compact_json = file.read()
        print("\nAperçu du fichier sample_submission.json:")
        print(compact_json[:500])  # Afficher les 500 premiers caractères
