# main.py
import json

import torch
from torch.utils.data import DataLoader
from load_data import load_datasets
from clean_transform_data import ARCDataset
from model import ImprovedCNN, train_model
from evaluate_model import evaluate_model
from generate_predictions import generate_predictions, write_predictions_to_json
import os


def main():
    # Charger les datasets
    print("Chargement des données...")
    (training_challenges, training_solutions, evaluation_challenges,
     evaluation_solutions, test_challenges) = load_datasets()
    print(f"Nombre de tâches d'entraînement: {len(training_challenges)}")
    print(f"Nombre de tâches de validation: {len(evaluation_challenges)}")
    print(f"Nombre de tâches de test: {len(test_challenges)}\n")

    # Préparer les datasets et les loaders
    train_dataset = ARCDataset(training_challenges, training_solutions)
    validation_dataset = ARCDataset(evaluation_challenges, evaluation_solutions)
    test_dataset = ARCDataset(test_challenges)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Initialiser le modèle
    model = ImprovedCNN()

    # Entraîner le modèle
    print("Entraînement du modèle...")
    train_model(model, train_loader, num_epochs=10, learning_rate=0.001)

    # Sauvegarder le modèle entraîné
    torch.save(model.state_dict(), 'trained_model.pth')
    print("Modèle entraîné et sauvegardé avec succès.\n")

    # Évaluer le modèle
    print("Évaluation du modèle sur les données de validation...")
    accuracy = evaluate_model(model, validation_loader)
    print(f"Précision du modèle: {accuracy * 100:.2f}%\n")

    # Générer les prédictions pour les données de test
    print("Génération des prédictions pour les données de test...")
    predictions = generate_predictions(model, test_loader)

    # Écrire les prédictions au format JSON requis
    submission_file_path = 'sample_submission.json'
    write_predictions_to_json(predictions, submission_file_path)
    print(f"Prédictions écrites dans le fichier {submission_file_path}.")

    # Afficher un aperçu du fichier de soumission
    if os.path.exists(submission_file_path):
        with open(submission_file_path, 'r') as file:
            sample_submission = json.load(file)
            print("\nAperçu du fichier sample_submission.json:")
            print(json.dumps(sample_submission, indent=4)[:500])  # Afficher les 500 premiers caractères


if __name__ == "__main__":
    main()
