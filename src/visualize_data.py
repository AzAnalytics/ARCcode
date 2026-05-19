# visualize_data.py
import matplotlib.pyplot as plt

def visualize_grid(grid, title="Grid"):
    """Visualize a grid using matplotlib."""
    plt.imshow(grid, cmap='tab20')
    plt.title(title)
    plt.colorbar()
    plt.show()

def visualize_example_task(task_id, training_challenges):
    """Visualize an example task from the training challenges."""
    example_task = training_challenges[task_id]
    example_input_grid = example_task['train'][0]['input']
    example_output_grid = example_task['train'][0]['output']

    visualize_grid(example_input_grid, title=f"Task {task_id}: Input Grid")
    visualize_grid(example_output_grid, title=f"Task {task_id}: Output Grid")

if __name__ == "__main__":
    from load_data import load_datasets
    training_challenges, _, _, _, _ = load_datasets()
    example_task_id = list(training_challenges.keys())[0]
    visualize_example_task(example_task_id, training_challenges)
