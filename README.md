# QUENCH Benchmark Implementation

## Overview

This repository contains an end-to-end implementation of the QUENCH benchmark, which evaluates multiple language models on entity prediction tasks using masked and causal language models. The implementation includes data processing, model inference, and evaluation with multiple performance metrics.

## Features

- **Adaptable Data Handling**: Parses YAML files from a Git repository and converts them into a standardized JSON format.
- **Multi-Model Compatibility**: Supports both masked and causal language models with automatic device mapping and memory-efficient FP16 precision.
- **Comprehensive Evaluation Suite**: Uses BLEU, ROUGE, and BERTScore to assess model performance.
- **Optimized for Kaggle Execution**: Progress bars, automatic memory management, and result persistence.
- **Flexible Deployment**: Works with various transformers and datasets; supports GPU acceleration.

## Requirements

Make sure you have the necessary dependencies installed:

```bash
pip install transformers datasets pyyaml rouge-score
```

## Models Used

This implementation benchmarks the following models:

- `distilbert-base-uncased`
- `huawei-noah/TinyBERT_General_4L_312D`
- `meta-llama/Llama-3B`
- `microsoft/Phi-4`
- `google/gemma-2-9b-it`

## Installation and Setup

1. **Prepare the Dataset**:

   - Upload your YAML files to Kaggle or store them in a local directory.
   - Set the correct dataset path in `load_quench_data()`.

2. **Run the Pipeline**:

   ```bash
   python main.py
   ```

   This will process the data, run model inference, and evaluate the results.

## Pipeline Breakdown

### Step 1: Data Preparation

- Parses YAML files and extracts masked questions, entities, rationales, and answers.
- Converts them into a standardized JSON format.

### Step 2: Model Loading

- Loads and configures masked and causal language models.
- Implements error handling for missing models or GPU constraints.

### Step 3: Prediction Handling

- Supports different architectures:
  - **Masked LMs** predict missing entities using `[MASK]` tokens.
  - **Causal LMs** generate missing entities from a prompt.

### Step 4: Evaluation

- Calculates **BLEU**, **ROUGE**, and **BERTScore** metrics.
- Handles Indic and non-Indic datasets separately if applicable.

### Step 5: Running the Benchmark

- Iterates over all models and processes dataset entries.
- Stores prediction outputs and aggregated evaluation metrics.

### Step 6: Result Analysis

- Outputs evaluation metrics for each model:
  ```bash
  BLEU: 0.65
  ROUGE-L: 0.78
  BERTScore F1: 0.85
  ```

## Performance Optimization

- Enable GPU in Kaggle settings for faster execution.
- Use **mixed precision (FP16)** to reduce memory usage.
- Implement batch processing to accelerate inference.
- Add caching mechanisms to store partial results in case of failures.

## Output Files

- `quench_processed.json`: Preprocessed dataset in JSON format.
- `quench_results.json`: Model predictions and evaluation scores.

## Troubleshooting

- **Model Loading Issues**: Ensure the model names are correct and available in Hugging Face.
- **Memory Errors**: Reduce batch sizes or switch to CPU execution.
- **Missing YAML Fields**: Update `load_quench_data()` to handle inconsistencies.

## Future Enhancements

- Implement parallel processing for faster execution.
- Expand model support to newer architectures.
- Add visualizations for result comparisons.



