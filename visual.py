import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def show_images(output_path: Path, batch_idx: int, dataset_name: str):
    """
    Function to load and display images (dirty, input, output) for a given batch index and dataset.

    Parameters:
        output_path (Path): The base path where the .npy files are saved.
        batch_idx (int): The index of the batch to display.
        dataset_name (str): The name of the dataset ('val' or 'test').
    """
    # Load the saved numpy arrays
    images_path = output_path / f"batch={batch_idx}_{dataset_name}_images.npy"
    generated_images_path = (
        output_path / f"batch={batch_idx}_{dataset_name}_generated_images.npy"
    )
    dirty_noisy_path = output_path / f"batch={batch_idx}_{dataset_name}_dirty_noisy.npy"

    images = np.load(images_path)
    generated_images = np.load(generated_images_path)
    dirty_noisy = np.load(dirty_noisy_path)

    # Display the images side by side
    num_images = len(images)

    for i in range(num_images):
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))

        ax[0].imshow(dirty_noisy[i])
        ax[0].set_title("Dirty/Noisy Image")
        ax[0].axis("off")

        ax[1].imshow(images[i])
        ax[1].set_title("Input Image")
        ax[1].axis("off")

        ax[2].imshow(generated_images[i])
        ax[2].set_title("Output Image")
        ax[2].axis("off")

        plt.show()


def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Display images from saved numpy arrays."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to the output directory where .npy files are saved.",
    )
    parser.add_argument(
        "--batch_idx", type=int, required=True, help="Index of the batch to display."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        choices=["val", "test"],
        default="test",
        help="Name of the dataset ('val' or 'test').",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    output_path = Path(args.output_path)
    show_images(output_path, args.batch_idx, args.dataset_name)
