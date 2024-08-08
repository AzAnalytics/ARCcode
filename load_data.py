# load_data.py
import json


def load_json(file_path):
    """Load JSON data from a file."""
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def load_datasets():
    """Load all datasets for the ARC Prize competition."""
    training_challenges = load_json('arc-agi_training_challenges.json')
    training_solutions = load_json('arc-agi_training_solutions.json')
    evaluation_challenges = load_json('arc-agi_evaluation_challenges.json')
    evaluation_solutions = load_json('arc-agi_evaluation_solutions.json')
    test_challenges = load_json('arc-agi_test_challenges.json')
    return training_challenges, training_solutions, evaluation_challenges, evaluation_solutions, test_challenges


if __name__ == "__main__":
    training_challenges, training_solutions, evaluation_challenges, evaluation_solutions, test_challenges = load_datasets()
    print(f"Loaded {len(training_challenges)} training tasks.")
