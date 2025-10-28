# 🎮 Valorant ML Prediction System# 🏆 Production ML System - League of Legends Match Prediction



A machine learning system that predicts Valorant player tiers and match outcomes using player statistics. Built with scikit-learn and Streamlit for an interactive web interface.[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)[![Kubernetes](https://img.shields.io/badge/kubernetes-ready-blue.svg)](https://kubernetes.io/)

![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

![Streamlit](https://img.shields.io/badge/streamlit-1.25+-red.svg)

![License](https://img.shields.io/badge/license-MIT-green.svg)A **production-grade machine learning system** for predicting League of Legends match outcomes, featuring advanced ML techniques, MLOps infrastructure, and enterprise-ready deployment capabilities.



## 🚀 Quick Start## 🎯 Performance Achievements



1. **Clone the repository**| Model | Accuracy | Cross-Validation | Status |

   ```bash|-------|----------|------------------|--------|

   git clone <your-repo-url>| **Advanced CatBoost** | **100.0%** | 99.86% ± 0.29% | ✅ Production |

   cd ML-capstone| **Advanced RandomForest** | **100.0%** | 99.57% ± 0.86% | ✅ Production |

   ```| Advanced LogisticRegression | 97.5% | 95.14% ± 2.36% | ✅ Production |

| Advanced ExtraTrees | 97.0% | 94.71% ± 1.07% | ✅ Production |

2. **Install dependencies**

   ```bash**🚀 Improvement: +42.5% accuracy gain** (from 57.5% baseline to 100% production)

   pip install -r requirements.txt

   ```## ⚡ Quick Start



3. **Train the model**### 1. Install Dependencies

   ```bash```bash

   python simple_ml_training.pypip install -r requirements.txt

   ``````



4. **Launch the web app**# ML Capstone — Valorant (work-in-progress)

   ```bash

   streamlit run ultimate_app.pyThis repository is a machine-learning capstone project that was originally built for League-of-Legends predictions and has been migrated toward a Valorant-focused UI and demo. The repo contains a production-ready training pipeline, multiple Streamlit demo apps, model modules, and dataset artifacts.

   ```

This README has been trimmed and updated to reflect the current project state and provide a concise developer quickstart.

5. **Open your browser** to `http://localhost:8501`

## Current status (2025-10-28)

## 📊 Features- UI: `ultimate_app.py` has been migrated to Valorant-style inputs and a heuristic prediction fallback.

- Training pipeline: `production_training.py` is a production-grade pipeline ready to run on match-level labeled data.

- **ML Model Training**: RandomForest classifier for tier prediction- Data: `data/processed/processed_data.csv` is match-level processed data (usable now); `data/raw/valorant_dataset_v3.csv` is player-aggregated Valorant data (no match outcome labels).

- **Interactive Web UI**: Streamlit-based prediction interface- Models: past model artifacts appear under `models/production` and `models/checkpoints`.

- **Real Data**: Trained on actual Valorant player statistics

- **Smart Fallback**: Graceful degradation to heuristic predictionsIf your goal is a working Valorant match-outcome model, you must provide match-level Valorant data with outcome labels. Meanwhile, the repo supports a demo training run on `data/processed/processed_data.csv` to validate training and wiring.

- **Model Persistence**: Trained models saved and reused

## Quickstart — developer (short)

## 🎯 Model Performance1) Create a fresh virtual env and install requirements:



- **Algorithm**: Random Forest Classifier```powershell

- **Training Accuracy**: 79%cd "C:\New folder\Codes\ML capstone"

- **Test Accuracy**: 48%python -m venv .venv

- **Features**: KD ratio, damage per match, kills per match, etc..\.venv\Scripts\Activate.ps1

pip install -r requirements.txt

## 📁 Project Structure```



```2) Run the production training pipeline (demo / proof-of-concept):

├── data/

│   └── raw/valorant_dataset_v3.csv    # Training dataset```powershell

├── models/python production_training.py

│   └── simple/                        # Trained model artifacts```

├── src/

│   ├── data/                          # Data processing utilitiesResults and reports will be written to `reports/` and model metadata to `models/`.

│   ├── models/                        # Model implementations

│   └── utils/                         # Helper functions3) Run the Streamlit app (demo UI):

├── tests/                             # Unit tests

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