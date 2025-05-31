# Toxicity Repair Experiments

This directory contains scripts for running toxicity repair experiments using OpenAI's GPT vision-language models. The primary goal is to evaluate the capability of these models to modify molecular structures to reduce toxicity while preserving therapeutic properties.

## Main Files

- `run_toxicity_repair.py`: The main script for running experiments across different toxicity datasets

## Requirements

- Python 3.8 or higher
- OpenAI Python package
- Properly prepared datasets in `data/Experimental_dataset/` directory, including:
  - Molecule JSON files (SMILES)
  - Molecular image files
  - Task prompt JSON files
  - Main repair prompt JSON file

## Usage

### Basic Usage

To run an experiment with the default settings:

```bash
python experiments/gpt/run_toxicity_repair.py
```

This will process all tasks using the default GPT-4.1-vision model.

### Run a Specific Task

To run the experiment for a specific task:

```bash
python experiments/gpt/run_toxicity_repair.py --task ames
```

Available tasks:
- ames
- carcinogens_lagunin
- clintox
- dili
- herg
- herg_central
- herg_karim
- ld50_zhu
- skin_reaction
- tox21
- toxcast

### Use a Different Model

To use a different OpenAI model:

```bash
python experiments/gpt/run_toxicity_repair.py --model gpt-4o-mini
```

### Limit the Number of Molecules

To limit the number of molecules processed per task:

```bash
python experiments/gpt/run_toxicity_repair.py --limit 10
```

### Process Specific Molecules

To process specific molecule IDs:

```bash
python experiments/gpt/run_toxicity_repair.py --task ames --molecule-ids 1 2 3
```

### Custom API Key

To use a custom API key:

```bash
python experiments/gpt/run_toxicity_repair.py --api-key YOUR_API_KEY
```

## Output

Results are saved in the `results/gpt/[model]/[task]/` directory structure. For each molecule, a JSON file is created containing:

- Original SMILES
- Task information
- Modified SMILES (1-3 candidates)
- Raw model response
- Metadata (timestamp, molecule ID, etc.)

Summary files are also created for each task and for the overall experiment.

## Example Workflow

1. Run a small test on one task with a limited number of molecules:
   ```bash
   python experiments/gpt/run_toxicity_repair.py --task ames --limit 5
   ```

2. Check results in `results/gpt/gpt-4.1-vision/ames/`

3. Run the full experiment across all tasks:
   ```bash
   python experiments/gpt/run_toxicity_repair.py --task all
   ```

## Notes

- The script includes retry logic and error handling to deal with API rate limits and other issues
- A short delay is added between molecule API calls to avoid rate limiting
- Make sure your OpenAI account has access to the model you're trying to use

## Toxicity Repair Experiments

This directory contains scripts for running toxicity repair experiments on molecules using various OpenAI models.

### Available Scripts

1. `run_toxicity_repair.py`: Runs toxicity repair experiments using OpenAI GPT models (e.g., GPT-4.1).

2. `run_inference_models.py`: Runs toxicity repair experiments using OpenAI reasoning models (o1, o3, o4-mini).

### Usage

#### GPT Models

```bash
python run_toxicity_repair.py --task [task_name|all] --model [gpt-model-name] --limit [number]
```

#### Reasoning Models (o1, o3, o4-mini)

```bash
python run_inference_models.py --task [task_name|all] --model [o1|o3|o4-mini] --reasoning-effort [low|medium|high] --limit [number]
```

Note: The `reasoning-effort` parameter is only applicable when using the `o3` model.

### Parameters

- `--task`: The toxicity task to run (default: all)
- `--model`: The model to use (default: depends on the script)
- `--reasoning-effort`: Reasoning effort level for o3 model (low, medium, high) (default: medium)
- `--api-key`: OpenAI API key (default: uses the key defined in the script)
- `--limit`: Maximum number of molecules to process per task (default: no limit)
- `--molecule-ids`: Specific molecule IDs to process (default: all molecules)

### Examples

Run all tasks with the o1 model:
```bash
python run_inference_models.py --model o1
```

Run the AMES test with the o3 model using high reasoning effort:
```bash
python run_inference_models.py --task ames --model o3 --reasoning-effort high
```

Process only the first 5 molecules for each task with the o4-mini model:
```bash
python run_inference_models.py --model o4-mini --limit 5
``` 