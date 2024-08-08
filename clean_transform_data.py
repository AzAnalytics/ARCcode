# clean_transform_data.py
import torch
from torch.utils.data import Dataset


def grid_to_tensor(grid, max_size=(30, 30)):
    """
    Convert a grid to a PyTorch tensor and pad it to a max_size.

    Parameters:
    - grid (list of list of int): The input grid.
    - max_size (tuple): The target size for padding (height, width).

    Returns:
    - tensor: A PyTorch tensor of size max_size.
    """
    tensor = torch.tensor(grid, dtype=torch.float32)
    # Initialize a new tensor filled with zeros with max_size
    padded_tensor = torch.zeros(max_size, dtype=torch.float32)
    # Copy the original tensor into the top-left corner of the padded tensor
    padded_tensor[:tensor.size(0), :tensor.size(1)] = tensor
    return padded_tensor


class ARCDataset(Dataset):
    """Custom Dataset for the ARC challenges."""

    def __init__(self, data_challenges, data_solutions=None, max_size=(30, 30)):
        """
        Initialize the dataset with challenges and solutions.

        Parameters:
        - data_challenges (dict): The input-output pairs for the tasks.
        - data_solutions (dict): The output solutions for validation (if available).
        - max_size (tuple): The target size for padding (height, width).
        """
        self.data_challenges = data_challenges
        self.data_solutions = data_solutions or {}
        self.tasks = list(data_challenges.keys())
        self.max_size = max_size

    def __len__(self):
        """Return the number of tasks in the dataset."""
        return len(self.tasks)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset at the specified index.

        Parameters:
        - idx (int): The index of the sample to retrieve.

        Returns:
        - tuple: (input_tensor, output_tensor) for training, or (input_tensor, task_id) for testing.
        """
        task_id = self.tasks[idx]
        task_data = self.data_challenges[task_id]

        # Select the first training pair (you can modify this to handle multiple pairs)
        input_grid = task_data['train'][0]['input']
        input_tensor = grid_to_tensor(input_grid, self.max_size)

        if self.data_solutions:
            output_grid = self.data_solutions[task_id][0]
            output_tensor = grid_to_tensor(output_grid, self.max_size)
            return input_tensor, output_tensor

        return input_tensor, task_id


if __name__ == "__main__":
    from load_data import load_datasets

    training_challenges, training_solutions, _, _, _ = load_datasets()

    # Initialize the dataset with padding
    train_dataset = ARCDataset(training_challenges, training_solutions)

    # Retrieve an example from the dataset
    input_tensor, output_tensor = train_dataset[0]

    print("Example input tensor shape:", input_tensor.shape)
    print("Example output tensor shape:", output_tensor.shape)
