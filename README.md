# 🎬 CineMatch-ML
### Personalized Movie Recommendation Engine

> ML-powered recommendation system combining Collaborative Filtering, Content-Based Filtering, and a Weighted Hybrid model — built on MovieLens 100K dataset.

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 👩‍💻 Author
**Meghana kamatam** — Data Science & ML Intern  
GitHub: [@MEGHANARA123456](https://github.com/MEGHANARA123456)

---

## 📌 About
CineMatch-ML is an enterprise-level recommendation engine that predicts and personalizes movie suggestions for users. It implements 5 different recommendation models and combines the best of collaborative and content-based approaches into a powerful hybrid system.

---

## 📁 Project Structure

```
CineMatch-ML/
│
├── data/
│   ├── raw/                    # Original MovieLens 100K dataset
│   └── processed/              # Cleaned and preprocessed CSVs
│
├── notebooks/
│   ├── content_based.ipynb      # TF-IDF content recommender
│   ├── hybrid_model.ipynb       # Hybrid SVD + TF-IDF model
│   └── evaluation.ipynb         # Full model comparison
│
├── src/
│   ├── collaborative/
│   │   ├── user_based.py           # User-based CF
│   │   ├── item_based.py           # Item-based CF
│   │   └── matrix_factorization.py # SVD and NMF models
│   ├── content_based/
│   │   └── tfidf_recommender.py    # TF-IDF content model
│   ├── hybrid/
│   │   └── hybrid_recommender.py   # Weighted hybrid model
│   └── utils/
│       ├── data_loader.py          # Data loading utilities
│       ├── preprocessor.py         # Data cleaning
│       └── evaluator.py            # RMSE, MAE, Precision@K
│
├── models/                         # Saved .pkl model files
├── app/
│   └── app.py                      # Streamlit dashboard
├── reports/
│   └── figures/                    # Charts and evaluation plots
├── tests/
│   ├── test_content_based.py       # Unit tests — content model
│   ├── test_hybrid.py              # Unit tests — hybrid model
│   └── test_collaborative.py       # Unit tests — CF models
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/MEGHANARA123456/CineMatch-ML.git
cd CineMatch-ML
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
Get [MovieLens 100K](https://grouplens.org/datasets/movielens/100k/) and place `u.data` and `u.item` in `data/raw/`.

### 4. Run notebooks in order
```
03_content_based → 04_hybrid_model → 05_evaluation
```

### 5. Launch Streamlit app
```bash
streamlit run app/app.py
```

### 6. Run unit tests
```bash
pytest tests/ -v
```

---

## 📊 Models Implemented

| Model | Type | RMSE | MAE |
|-------|------|------|-----|
| User-based CF | Collaborative Filtering | ~1.02 | ~0.81 |
| Item-based CF | Collaborative Filtering | ~0.98 | ~0.78 |
| SVD | Matrix Factorization | ~0.93 | ~0.73 |
| NMF | Matrix Factorization | ~0.96 | ~0.75 |
| TF-IDF | Content-Based | — | — |
| **Hybrid (SVD + TF-IDF)** | **Combined** | **~0.91** | **~0.72** |

> ✅ **Hybrid model performs best** — combines personalization from SVD with genre-based discovery from TF-IDF.

---

## 📈 Evaluation Metrics
- **RMSE** — Root Mean Square Error (lower is better)
- **MAE** — Mean Absolute Error (lower is better)
- **Precision@10** — Fraction of top-10 recommendations that are relevant
- **Recall@10** — Fraction of relevant items that appear in top-10

---

## 🔬 Key Features
- ✅ 5 recommendation models implemented and compared
- ✅ Weighted hybrid model with tunable alpha parameter
- ✅ User feedback loop simulation
- ✅ 23 unit tests — all passing
- ✅ Interactive Streamlit dashboard
- ✅ Full model evaluation with comparison charts

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.10 | Core language |
| Pandas & NumPy | Data processing |
| Scikit-learn | TF-IDF, cosine similarity |
| Scikit-surprise | SVD, NMF, collaborative filtering |
| Matplotlib & Seaborn | Visualizations |
| Streamlit | Interactive dashboard |
| Pytest | Unit testing |
| Joblib | Model serialization |

---

## 📅 Development Timeline
| Week | Tasks |
|------|-------|
| Week 1 | Data loading, EDA, preprocessing |
| Week 2 | Collaborative filtering models |
| Week 3 | Content-based + Hybrid + Evaluation |
| Week 4 | Streamlit dashboard + Final report |

---

## 📄 License
MIT License — feel free to use and modify.
