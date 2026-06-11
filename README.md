# Justifying the Family: A Semantic Analysis of Rhetorical Trends

## Allinn Chen, Rean Du, Tyler Tran, and David Wang

### How to Get Started

#### 1. Clone the Repository

```bash
git clone https://github.com/all-ch/capstone-project
cd ./capstone-project
```

#### 2. Install `uv`

**macOS/Linux**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows**

```
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

#### 3. Installing Dependencies

```bash
uv sync
```

### How to Run

#### 1. Load Virtual Environment

```bash
source .venv/bin/activate
```

#### 2. Open JupyterLab

```bash
uv run jupyter lab
```

#### 3. Run the Script

Go to top menu and select **Run > Run All Cells**

### File Descriptions

#### Repository Structure

```text
├── capstone-project.Rproj
├── data
│   ├── anchors
│   │   ├── [topic]_[sentiment]_anchor_sentences.csv
│   ├── processed
│   │   ├── speeches.csv
│   │   └── table.parquet
│   └── raw
│       └── data.xlsx
├── main.html
├── main.ipynb
├── main.py
├── pyproject.toml
├── python
│   ├── embeddings.py
│   ├── models.py
│   ├── pca.py
│   └── recompute.py
├── R
│   └── preprocessing.qmd
├── README.md
├── tests
│   └── test_anchors.py
└── uv.lock
```

#### Data

- `data/raw/data.xlsx`: the raw dataset provided by Dr. Jeffrey Swindle.
- `data/processed/speeches.csv`: the final version of data containing cache.
- `data/processed/table.parquet`: compressed version of `speeches.csv` used for calculations.
- `data/anchors/`: contains anchor sentences for three topics (religious, political, and scientific justification) across two sentiment polarities (positive and negative).

#### Utilities

- `python/embeddings.py`: helper functions for embedding conversion and calculations, model initialization, and cosine similarity analysis.
- `python/models.py`: helper functions for fitting and plotting Gaussian Mixture and Linear Regression models.
- `python/pca.py`: helper functions for fitting and plotting Principal Component Analysis.
- `python/recompute`: helper functions to recompute all models and similarity scores.

#### Scripts

- `R/preprocessing.qmd`: script that filters raw data into **speech_id**, **speaker_id**, **speaker**, **year**, and **speech**.
- `tests/test_anchors.py`: script to compare performance of the religious justification semantic topic axis across five categories of test cases.
- `main.py`: script to save all plots to the `outputs` folder and optionally select to recompute all values. (require the creation of the `outputs` folder or selection of output directory)
- `main.ipynb`: script to show all plots.
- `main.html`: web output of `main.ipynb`.
