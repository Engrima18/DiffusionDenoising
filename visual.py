import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf


def standardize_image(image, min_val, max_val):
    """
    Standardize an image using the provided minimum and maximum values.

    Parameters:
        image (np.ndarray): The image to standardize.
        min_val (float): The minimum value for scaling.
        max_val (float): The maximum value for scaling.

    Returns:
        np.ndarray: The standardized image.
    """
    standardized_image = (image - min_val) / (max_val - min_val)
    return standardized_image


def show_images_and_save(
    output_path: Path,
    plot_path: Path,
    batch_idx: int,
    dataset_name: str,
    model_name: str,
):
    """
    Function to load and display images (dirty, input, output) for a given batch index and dataset,
    and save the displayed images in the 'plots' directory.

    Parameters:
        output_path (Path): The base path where the .npy files are saved.
        batch_idx (int): The index of the batch to display.
        dataset_name (str): The name of the dataset ('val' or 'test').
        model_name (str): The name of the model used, added to the plot filenames.
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

    # Find the min and max values from the input images for standardization
    min_val = np.min(images, axis=(1, 2, 3))
    max_val = np.max(images, axis=(1, 2, 3))

    # Create plots directory if it doesn't exist
    plot_path.mkdir(exist_ok=True)

    # Display the images side by side and save them
    num_images = len(images)

    for i in range(num_images):
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))

        # Standardize images using the min and max values from the input images
        dirty_image = dirty_noisy[i]
        input_image = images[i]
        output_image = generated_images[i]

        ax[0].imshow(dirty_image)
        ax[0].set_title("Dirty/Noisy Image")
        ax[0].axis("off")

        ax[1].imshow(input_image)
        ax[1].set_title("Input Image")
        ax[1].axis("off")

        ax[2].imshow(output_image, vmin=min_val[i], vmax=max_val[i])
        ax[2].set_title("Output Image")
        ax[2].axis("off")

        # Save the figure
        plot_filename = plot_path / f"{images_path.stem}_{model_name}_img{i}.png"
        plt.savefig(plot_filename)
        plt.close(fig)


@hydra.main(config_path="configs", config_name="visualization", version_base="1.3")
def main(cfg: DictConfig):
    """
    Main function to run the image display and saving script.

    Parameters:
        cfg (DictConfig): Configuration loaded by Hydra.
    """
    path = Path(f"{cfg.output_path}\\{cfg.project_name}")
    model_name = f"power{cfg.power}_{cfg.loss}_{cfg.generation_mode}"
    output_path = path / model_name
    plot_path = Path(cfg.plot_path)
    plot_path = plot_path / cfg.project_name
    show_images_and_save(
        output_path, plot_path, cfg.batch_idx, cfg.dataset_name, model_name
    )


if __name__ == "__main__":
    main()
