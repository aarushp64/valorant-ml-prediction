# 🎮 Valorant ML Prediction System# 🎮 Valorant ML Prediction System# 🏆 Production ML System - League of Legends Match Prediction



A machine learning system that predicts Valorant player tiers and match outcomes using player statistics. Built with scikit-learn and Streamlit for an interactive web interface.



![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)A machine learning system that predicts Valorant player tiers and match outcomes using player statistics. Built with scikit-learn and Streamlit for an interactive web interface.[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)

![Streamlit](https://img.shields.io/badge/streamlit-1.25+-red.svg)[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)

![License](https://img.shields.io/badge/license-MIT-green.svg)

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)[![Kubernetes](https://img.shields.io/badge/kubernetes-ready-blue.svg)](https://kubernetes.io/)

## 🚀 Quick Start

![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

1. **Clone the repository**

   ```bash![Streamlit](https://img.shields.io/badge/streamlit-1.25+-red.svg)

   git clone https://github.com/aarushp64/valorant-ml-prediction.git

   cd valorant-ml-prediction![License](https://img.shields.io/badge/license-MIT-green.svg)A **production-grade machine learning system** for predicting League of Legends match outcomes, featuring advanced ML techniques, MLOps infrastructure, and enterprise-ready deployment capabilities.

   ```



2. **Install dependencies**

   ```bash## 🚀 Quick Start## 🎯 Performance Achievements

   pip install -r requirements.txt

   ```



3. **Train the model**1. **Clone the repository**| Model | Accuracy | Cross-Validation | Status |

   ```bash

   python simple_ml_training.py   ```bash|-------|----------|------------------|--------|

   ```

   git clone <your-repo-url>| **Advanced CatBoost** | **100.0%** | 99.86% ± 0.29% | ✅ Production |

4. **Launch the web app**

   ```bash   cd ML-capstone| **Advanced RandomForest** | **100.0%** | 99.57% ± 0.86% | ✅ Production |

   streamlit run ultimate_app.py

   ```   ```| Advanced LogisticRegression | 97.5% | 95.14% ± 2.36% | ✅ Production |



5. **Open your browser** to `http://localhost:8501`| Advanced ExtraTrees | 97.0% | 94.71% ± 1.07% | ✅ Production |



## 📊 Features2. **Install dependencies**



- **ML Model Training**: RandomForest classifier for tier prediction   ```bash**🚀 Improvement: +42.5% accuracy gain** (from 57.5% baseline to 100% production)

- **Interactive Web UI**: Streamlit-based prediction interface

- **Real Data**: Trained on actual Valorant player statistics   pip install -r requirements.txt

- **Smart Fallback**: Graceful degradation to heuristic predictions

- **Model Persistence**: Trained models saved and reused   ```## ⚡ Quick Start



## 🎯 Model Performance



- **Algorithm**: Random Forest Classifier3. **Train the model**### 1. Install Dependencies

- **Training Accuracy**: 79%

- **Test Accuracy**: 48%   ```bash```bash

- **Dataset**: 2,694 Valorant player records

- **Features**: KD ratio, damage per match, kills per match, etc.   python simple_ml_training.pypip install -r requirements.txt



## 📁 Project Structure   ``````



```

├── data/

│   └── raw/valorant_dataset_v3.csv    # Training dataset4. **Launch the web app**# ML Capstone — Valorant (work-in-progress)

├── models/

│   └── simple/                        # Trained model artifacts   ```bash

├── src/

│   ├── data/                          # Data processing utilities   streamlit run ultimate_app.pyThis repository is a machine-learning capstone project that was originally built for League-of-Legends predictions and has been migrated toward a Valorant-focused UI and demo. The repo contains a production-ready training pipeline, multiple Streamlit demo apps, model modules, and dataset artifacts.

│   ├── models/                        # Model implementations

│   └── utils/                         # Helper functions   ```

├── tests/                             # Unit tests

├── simple_ml_training.py              # Training scriptThis README has been trimmed and updated to reflect the current project state and provide a concise developer quickstart.

├── ultimate_app.py                    # Streamlit web app

└── requirements.txt                   # Dependencies5. **Open your browser** to `http://localhost:8501`

```

## Current status (2025-10-28)

## 🔧 Usage

## 📊 Features- UI: `ultimate_app.py` has been migrated to Valorant-style inputs and a heuristic prediction fallback.

### Training a New Model

```bash- Training pipeline: `production_training.py` is a production-grade pipeline ready to run on match-level labeled data.

python simple_ml_training.py

```- **ML Model Training**: RandomForest classifier for tier prediction- Data: `data/processed/processed_data.csv` is match-level processed data (usable now); `data/raw/valorant_dataset_v3.csv` is player-aggregated Valorant data (no match outcome labels).



### Running the Web Interface- **Interactive Web UI**: Streamlit-based prediction interface- Models: past model artifacts appear under `models/production` and `models/checkpoints`.

```bash

streamlit run ultimate_app.py- **Real Data**: Trained on actual Valorant player statistics

```

- **Smart Fallback**: Graceful degradation to heuristic predictionsIf your goal is a working Valorant match-outcome model, you must provide match-level Valorant data with outcome labels. Meanwhile, the repo supports a demo training run on `data/processed/processed_data.csv` to validate training and wiring.

### Running Tests

```bash- **Model Persistence**: Trained models saved and reused

pytest tests/

```## Quickstart — developer (short)



## 📈 Model Details## 🎯 Model Performance1) Create a fresh virtual env and install requirements:



The system uses a **Random Forest Classifier** trained on:

- Player kill/death statistics

- Damage dealt metrics- **Algorithm**: Random Forest Classifier```powershell

- Combat score data

- Match performance indicators- **Training Accuracy**: 79%cd "C:\New folder\Codes\ML capstone"



Features are engineered to create:- **Test Accuracy**: 48%python -m venv .venv

- KD ratio (kills/deaths)

- Damage per match- **Features**: KD ratio, damage per match, kills per match, etc..\.venv\Scripts\Activate.ps1

- Kills per match

- Combat efficiency metricspip install -r requirements.txt



## 🤝 Contributing## 📁 Project Structure```



1. Fork the repository

2. Create a feature branch

3. Make your changes```2) Run the production training pipeline (demo / proof-of-concept):

4. Add tests

5. Submit a pull request├── data/



## 📄 License│   └── raw/valorant_dataset_v3.csv    # Training dataset```powershell



This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.├── models/python production_training.py



## 🎮 About│   └── simple/                        # Trained model artifacts```



This project demonstrates end-to-end machine learning for gaming analytics, showcasing:├── src/

- Data preprocessing and feature engineering

- Model training and evaluation│   ├── data/                          # Data processing utilitiesResults and reports will be written to `reports/` and model metadata to `models/`.

- Web-based deployment with Streamlit

- Code organization and testing best practices│   ├── models/                        # Model implementations



---│   └── utils/                         # Helper functions3) Run the Streamlit app (demo UI):



**Repository**: https://github.com/aarushp64/valorant-ml-prediction  ├── tests/                             # Unit tests

**Live Demo**: Run locally with `streamlit run ultimate_app.py`
├── simple_ml_training.py              # Training script```powershell

├── ultimate_app.py                    # Streamlit web appstreamlit run ultimate_app.py

└── requirements.txt                   # Dependencies```

```

4) Run unit tests (if desired):

## 🔧 Usage

```powershell

### Training a New Modelpytest -q

```bash```

python simple_ml_training.py

```## Important data notes

- `data/processed/processed_data.csv` — match-level training data (used by the pipeline). Inspect the `win` or `team1_win` column — the pipeline expects a target column.

### Running the Web Interface- `data/raw/valorant_dataset_v3.csv` — player-aggregated Valorant dataset (no per-match outcomes). Useful for sampling, EDA, and demo team generation, but not for supervised match-outcome training.

```bash

streamlit run ultimate_app.py## Where things are

```- Streamlit UIs: `ultimate_app.py`, `simple_app.py`, `app.py` (legacy)

- Training pipeline: `production_training.py`

### Running Tests- Models: `models/production/` and `models/checkpoints/`

```bash- Data: `data/raw/` and `data/processed/`

pytest tests/- Tests: `tests/`

```

## Next recommended steps

## 📈 Model Details1. Run a demo training on `data/processed/processed_data.csv` to validate pipeline and produce artifacts.

2. Wire the saved model into `ultimate_app.py` (toggle trained model vs heuristic). I can do this for you.

The system uses a **Random Forest Classifier** trained on:3. If you want a real Valorant match-outcome model, provide match-level Valorant logs/CSV with labeled outcomes.

- Player kill/death statistics

- Damage dealt metrics## Contribution & dev notes

- Combat score data- Use the todo list in repo root to track high-level tasks. The codebase contains in-repo notebooks and scripts used during development.

- Match performance indicators

## Contacts & support

Features are engineered to create:- For issues or to provide new datasets, attach them to this repo or place them under `data/raw/` and notify the team.

- KD ratio (kills/deaths)

- Damage per match---

- Kills per match

- Combat efficiency metricsMinimal cleanup applied: README trimmed and updated to reflect current status and immediate next steps.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🎮 About

This project demonstrates end-to-end machine learning for gaming analytics, showcasing:
- Data preprocessing and feature engineering
- Model training and evaluation
- Web-based deployment with Streamlit
- Code organization and testing best practices