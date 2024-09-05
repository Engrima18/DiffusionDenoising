# Diffusion models for denoising astronomical images

## Prerequisites

- [Conda](https://docs.conda.io/en/latest/miniconda.html) installed on your machine.
- Recommended: Miniconda or Anaconda distribution.

## Setting Up the Environment

To set up the project environment and install all necessary dependencies, follow these steps:

### 1. Clone the repository

First, clone the repository to your local machine:

```bash
git clone https://github.com/Engrima18/DiffusionDenoising
cd DiffusionDenoising
```

### 2. Install Anaconda Project (if not already installed)

Anaconda Project is used to manage the environment setup.

```bash
conda install anaconda-project
```

### 3. Create the environment

Once inside the project folder, create the environment using the provided anaconda-project.yaml file:

```bash
anaconda-project prepare
```

## Running the Project

There are several commands available in this project, including running different scripts like training, denoising, and plotting.

### 1. Training Script

To run the `train.py` script and train a Denoising Diffusion model with all the specifications from the configuration file (you choose the dataset).

```bash
anaconda-project run train
```

### 2. Denoising Script

To run the `denoising.py` script and starting the denoising process starting from certain input noisy images. In practice with this command you will use some diffusion model checkpoint to generate new images starting from noisy test images.

```bash
anaconda-project run denoising
```

### 3. Plotting script

To run the `visual.py` script and plot some results from the above denoising/generation process.

```bash
anaconda-project run plot
```
