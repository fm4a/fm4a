# AI Foundation Models for the Atmosphere Workshop: Hands-on Sessions

This repository contains hands-on workshop materials for the AI Foundation Models for the Atmosphere workshop, featuring Jupyter notebooks for three different sessions focused on applications of the Prithvi-WxC AI FM for weather and climate modeling.

## Repository Structure

- **session_1/**: Introduction of the PrithviWxC model
  - `PrithviWxC_input_data.ipynb`: Input data structure and masked predictions.
  - `PrithviWxC_rollout.ipynb`: P
  - `PrithviWxC_tc_forecast.ipynb`: Tropical cyclone forecasting

- **session_2/**: Downscaling using the Prithvi-WxC FM
  - `eccc_downscaling_finetune.ipynb`:  Fine-tuning downscaling models
  - `eccc_downscaling_inference.ipynb`: Running inference with trained models

- **session_3/**: Precipitation Forecasting using the Prithvi-WxC FM
  - `PrithviPrecip_introduction.ipynb`: Introduction to the Prithvi Precip forecasting model.

## Environment Setup

### Prerequisites
- conda (Anaconda or Miniconda)
- wget (for downloading model weights)

### Quick Setup

Run the provided setup script to automatically create the conda environment and download required model weights:

```bash
./setup.sh
```

The setup script will:
1. Create a conda environment named `fm4a` with Python 3.12
2. Install all required packages from `requirements.txt`
3. Download Prithvi Weather & Climate model weights
4. Register the environment as a Jupyter kernel

### Manual Setup (Alternative)

If you prefer to set up the environment manually:

```bash
# Create conda environment
conda create -n fm4a python=3.12 -y

# Activate environment
conda activate fm4a

# Install UV package manager
pip install uv

# Install requirements
uv pip install -r requirements.txt

# Register Jupyter kernel
python -m ipykernel install --user --name fm4a --display-name fm4a

# Create data directory and download weights
mkdir -p data/weights/
wget -O data/weights/prithvi.wxc.rollout.600m.v1.pt https://www.nsstc.uah.edu/data/sujit.roy/demo/consolidated.pth --no-check-certificate
```

## Running the Notebooks

> **NOTE:** To run the examples you will need a GPU with at least 48 GB of VRAM to run most examples. If you have no GPU available, running the examples should still be possible but will take longer. To do so, set the ``device`` variables in the notebooks to ``cpu``.

### Start Jupyter Lab

After setting up the environment, start Jupyter Lab:

```bash
# Activate the environment
conda activate fm4a

# Start Jupyter Lab
jupyter lab
```

### Running Individual Sessions

Navigate to the desired session folder and open the notebooks:

1. **Session 1 (PrithviWxC)**: Start with `session_1/` notebooks for weather and climate modeling
2. **Session 2 (ECCC Downscaling)**: Use `session_2/` notebooks for downscaling workflows  
3. **Session 3 (Prithvi Precipitation)**: Explore `session_3/` for precipitation modeling

