# Adaptive Machine Learning Framework for Debris Flow Monitoring

This repository contains the code for the paper:

**Chang, Jui-Ming and Zhou, Qi and Tang, Hui and Turowski, Jens M. and Ko, Ko, "Adaptive Machine Learning Framework for Debris Flow Monitoring in Nonstationary Environments in Illgraben, Switzerland"**

Available at SSRN: https://ssrn.com/abstract=5736972 or http://dx.doi.org/10.2139/ssrn.5736972

## Repository Structure

```
├── debris_flow.ipynb          # Main notebook with complete analysis and results
├── Scenario_I.py              # Scenario I: Adaptive threshold model
├── Scenario_II.py             # Scenario II: Global threshold calculation
├── Scenario_IV.py             # Scenario IV: Advanced adaptive model
├── no_lstm.py                 # Ablation study: Model without LSTM
├── no_rf.py                   # Ablation study: Model without Random Forest
├── no_xgb.py                  # Ablation study: Model without XGBoost
├── requirements.txt           # Required Python packages
├── README.md                  # This file
└── .gitignore                 # Git ignore file
```

## Overview

This project presents an adaptive machine learning framework for debris flow monitoring in nonstationary environments, with a case study in Illgraben, Switzerland. The framework integrates multiple machine learning models and adaptive threshold mechanisms to improve debris flow detection accuracy.

## Features

- **Adaptive Threshold Framework**: Dynamic adjustment of detection thresholds based on environmental changes
- **Ensemble Learning**: Combination of XGBoost, Random Forest, and LSTM models
- **Multiple Scenarios**: Four different implementation scenarios for comparison
- **Ablation Studies**: Analysis of individual model contributions
- **Drift Detection**: Implementation of concept drift detection mechanisms

## Requirements

This project requires Python 3.8+ and the following packages:

- numpy==1.26.4
- pandas==2.2.2
- torch==2.3.0
- xgboost==2.0.3
- scikit-learn
- optuna==3.6.1
- alibi-detect==0.11.4
- matplotlib
- seaborn

For a complete list, see `requirements.txt`.

## Installation

1. Clone this repository:
```bash
git clone https://github.com/YOUR_USERNAME/debris-flow-adaptive-ml.git
cd debris-flow-adaptive-ml
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Main Analysis

The complete analysis and results are available in `debris_flow.ipynb`. This notebook contains:
- Data preprocessing
- Model training and evaluation
- Visualization of results
- Performance metrics

To run the notebook:
```bash
jupyter notebook debris_flow.ipynb
```

Or use Google Colab (recommended for GPU support):
1. Upload the notebook to Google Drive
2. Open with Google Colab
3. Follow the instructions in the notebook to mount your drive and load data

### Scenario Scripts

Each scenario can be run independently:

**Scenario I** - Adaptive threshold with full feature set:
```bash
python Scenario_I.py
```

**Scenario II** - Global threshold approach:
```bash
python Scenario_II.py
```

**Scenario IV** - Advanced adaptive model:
```bash
python Scenario_IV.py
```

### Ablation Studies

To evaluate the contribution of individual models:

```bash
python no_lstm.py   # Without LSTM component
python no_rf.py     # Without Random Forest component
python no_xgb.py    # Without XGBoost component
```

## Data

**Note**: The dataset is not included in this repository due to size constraints. 

The dataset can be downloaded from Zenodo:

**[Dataset] Zhou, Q. (2025, March). Supporting material for "Enhancing debris flow warning via machine learning feature reduction and model selection". Zenodo.**

DOI: https://doi.org/10.5281/zenodo.15020368

### Data Files

Expected data format:
- Training data: ILL12_2017_2018.csv, ILL13_2017_2018.csv, ILL18_2017_2018.csv
- Validation data: ILL12_2019.csv, ILL13_2019.csv, ILL18_2019.csv
- Test data: test_ILL12_2020.csv, test_ILL13_2020.csv, test_ILL18_2020.csv

Please download the data from the Zenodo repository and place the CSV files in your working directory before running the code.

## Models

The framework employs an ensemble of three models:

1. **XGBoost**: Gradient boosting for handling complex feature interactions
2. **Random Forest**: Robust classification with feature importance analysis
3. **LSTM**: Sequential pattern recognition for temporal dependencies

## Citation

If you use this code in your research, please cite:

```bibtex
@article{chang2024adaptive,
  title={Adaptive Machine Learning Framework for Debris Flow Monitoring in Nonstationary Environments in Illgraben, Switzerland},
  author={Chang, Jui-Ming and Zhou, Qi and Tang, Hui and Turowski, Jens M. and Ko, Ko},
  journal={Available at SSRN 5736972},
  year={2024},
  doi={10.2139/ssrn.5736972}
}
```

## Authors

- Jui-Ming Chang
- Qi Zhou
- Hui Tang
- Jens M. Turowski
- Ko Ko

## License

This code is provided for academic and research purposes with a citation requirement. See [LICENSE](LICENSE) for details.

**If you use this code, you must cite the preprint article.**

## Acknowledgments

This research was conducted with data from the Illgraben debris flow monitoring station in Switzerland.

## Contact

For questions or collaboration inquiries, please contact the corresponding author through the SSRN page.

## Notes

- Scenario III is not included as a separate script as it uses the global threshold calculation from Scenario II
- All scripts are configured for reproducibility with fixed random seeds (RANDOM_STATE = 42)
- GPU support is recommended for faster training (tested with CUDA-compatible devices)
