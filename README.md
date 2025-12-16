# 🎙️ Voice Analysis Evaluation Pipeline

## Overview
This project evaluates **Gemini / OpenRouter audio understanding** by:

1. Running Gemini analysis on each audio file  
2. Comparing model predictions against ground truth labels

---

## Prerequisites

### Install Python 3.10+
Verify your Python version:

```bash
python --version
```

### Export API Keys

#### OpenRouter API Key
```powershell
$env:OPENROUTER_API_KEY="your-openrouter-key-here"
```

---

## Project Structure

```text
voice_analysis/
│
├── data/
│   └── raw/                    # Input .opus audio files
│
├── features.json               # Feature schema definitions
├── labels.json                 # Ground truth labels for evaluation
├── run_audio_eval.py           # Main evaluation script
├── results_gemini.csv          # Output results (generated)
└── README.md                   # Project documentation
```


## Run Gemini Analysis

### Run with Default Gemini Model
```bash
python run_audio_eval.py
```

### Run with a Specific Model Variant
```bash
python run_audio_eval.py --variant gemini-3-pro-preview
```

```bash
python run_audio_eval.py --variant gemini-2.5-flash
```

### Output
Results will be saved to:

```text
results_gemini.csv
```
