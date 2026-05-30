# ML-Resources

## Contents

- [Prerequisites](#prerequisites)
- [Foundations](#foundations)
- [Data Tooling (Pandas, NumPy)](#data-tooling-pandas-numpy)
- [Classical Machine Learning (scikit-learn)](#classical-machine-learning-scikit-learn)
- [Deep Learning](#deep-learning)
- [Specializations](#specializations)
- [Coding & Implementation](#coding--implementation)
- [Practice & Datasets](#practice--datasets)
- [Papers](#papers)
- [Communities & Staying Current](#communities--staying-current)

## Prerequisites

Math and programming background that makes the rest go smoothly.

- **Linear Algebra** — [3Blue1Brown: Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) (visual intuition), [MIT 18.06 (Strang)](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/)
- **Calculus** — [3Blue1Brown: Essence of Calculus](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr)
- **Probability & Statistics** — [Harvard Stat 110 (Blitzstein)](https://projects.iq.harvard.edu/stat110/home), [Seeing Theory](https://seeing-theory.brown.edu/)
- **Python** — [Python for Everybody](https://www.py4e.com/), NumPy/Pandas via [the official tutorials](https://numpy.org/learn/)

## Foundations

Start-here courses that cover the core ideas end to end.

- [Andrew Ng — Machine Learning Specialization (Coursera)](https://www.coursera.org/specializations/machine-learning-introduction) — the standard on-ramp
- [fast.ai — Practical Deep Learning for Coders](https://course.fast.ai/) — code-first, top-down approach
- [CS229: Machine Learning (Stanford)](https://cs229.stanford.edu/) — the rigorous version, full notes and problem sets online
- [Google ML Crash Course](https://developers.google.com/machine-learning/crash-course) — quick, interactive

## Data Tooling (Pandas, NumPy)

The libraries you'll use to load, clean, and explore data before any model touches it.

- [NumPy: the absolute basics](https://numpy.org/doc/stable/user/absolute_beginners.html) — array fundamentals
- [100 NumPy Exercises](https://github.com/rougier/numpy-100) — drills from easy to hard
- [Pandas: 10 Minutes to Pandas](https://pandas.pydata.org/docs/user_guide/10min.html) — official quick start
- [Pandas Getting Started Tutorials](https://pandas.pydata.org/docs/getting_started/intro_tutorials/index.html) — task-oriented walkthroughs
- [pandas Cookbook (Julia Evans)](https://github.com/jvns/pandas-cookbook) — real-world recipes
- [Effective Pandas (Matt Harrison)](https://github.com/mattharrison/effective_pandas) — idiomatic, chainable style
- [Kaggle: Pandas micro-course](https://www.kaggle.com/learn/pandas) — short, hands-on
- [Python Data Science Handbook (VanderPlas)](https://jakevdp.github.io/PythonDataScienceHandbook/) — free book covering NumPy, Pandas, and Matplotlib

## Classical Machine Learning (scikit-learn)

Models and methods that predate (and still beat) deep nets on tabular data.

- [Hands-On Machine Learning (Géron)](https://github.com/ageron/handson-ml3) — book + notebooks, scikit-learn and Keras
- [An Introduction to Statistical Learning (ISLP)](https://www.statlearning.com/) — free PDF, Python labs
- [The Elements of Statistical Learning](https://hastie.su.domains/ElemStatLearn/) — the deeper reference

### scikit-learn

- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html) — surprisingly good as a learning resource
- [Getting Started](https://scikit-learn.org/stable/getting_started.html) — fit/predict/pipelines in one page
- [Tutorials](https://scikit-learn.org/stable/tutorial/index.html) — official guided intros
- [Examples Gallery](https://scikit-learn.org/stable/auto_examples/index.html) — runnable code for nearly every estimator
- [Choosing the right estimator (cheat sheet)](https://scikit-learn.org/stable/machine_learning_map.html) — flowchart for picking a model
- [Kaggle: Intro to Machine Learning](https://www.kaggle.com/learn/intro-to-machine-learning) — sklearn-based, beginner friendly

## Deep Learning

- [Deep Learning Book (Goodfellow, Bengio, Courville)](https://www.deeplearningbook.org/) — free, theory-heavy
- [Neural Networks: Zero to Hero (Karpathy)](https://karpathy.ai/zero-to-hero.html) — build backprop, makemore, and a GPT from scratch
- [CS231n: CNNs for Visual Recognition (Stanford)](https://cs231n.github.io/) — notes are excellent even standalone
- [Dive into Deep Learning (d2l.ai)](https://d2l.ai/) — interactive book with PyTorch/TF/JAX code

## Specializations

- **NLP / LLMs** — [CS224n (Stanford)](https://web.stanford.edu/class/cs224n/), [Hugging Face NLP Course](https://huggingface.co/learn/nlp-course)
- **Computer Vision** — [CS231n](https://cs231n.github.io/)
- **Reinforcement Learning** — [Spinning Up in Deep RL (OpenAI)](https://spinningup.openai.com/), [Sutton & Barto (free PDF)](http://incompleteideas.net/book/the-book.html)
- **MLOps / Production** — [Made With ML](https://madewithml.com/), [Full Stack Deep Learning](https://fullstackdeeplearning.com/)

## Coding & Implementation

For going beyond `import` and building things yourself.

- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [nn-from-scratch (Karpathy micrograd)](https://github.com/karpathy/micrograd) — autograd in ~100 lines
- [ML algorithms from scratch (Erik Linder-Norén)](https://github.com/eriklindernoren/ML-From-Scratch)
- [labml.ai Annotated Implementations](https://nn.labml.ai/) — papers reimplemented with side-by-side notes

## Practice & Datasets

- [Kaggle](https://www.kaggle.com/) — competitions, datasets, and free notebooks
- [Google Colab](https://colab.research.google.com/) — free GPUs for experiments
- [Papers with Code](https://paperswithcode.com/) — SOTA results linked to implementations
- [UCI ML Repository](https://archive.ics.uci.edu/) — classic small datasets
- [Hugging Face Datasets](https://huggingface.co/datasets)

## Papers

- [arXiv Sanity / arxiv.org cs.LG](https://arxiv.org/list/cs.LG/recent)
- [The Illustrated Transformer (Jay Alammar)](https://jalammar.github.io/illustrated-transformer/) — best intro to attention
- Foundational reads: *Attention Is All You Need*, *Deep Residual Learning (ResNet)*, *Adam optimizer*, *Dropout*, *Batch Normalization*

## Communities & Staying Current

- [r/MachineLearning](https://www.reddit.com/r/MachineLearning/)
- [Distill.pub](https://distill.pub/) — visual explanations (archived but timeless)
- Newsletters: The Batch (deeplearning.ai), Import AI


```python
"""
THE 5 PHASES (unchanged in spirit):
  1. Orient  2. Split(+CV)  3. Bake-off  4. Diagnose  5. Finalize
============================================================================
"""

# %% ==========================================================================
# PHASE 1 — ORIENT
# =============================================================================
import numpy as np
import pandas as pd

train = pd.read_csv("data/train.csv")   # <-- adjust path/filename

print(train.shape)
print(train.isna().sum())
print(train.describe())

# SMALL-DATA NOTES:
#  - With ~500 rows, eyeball outliers carefully (describe min/max vs quartiles).
#    One leverage point matters more here than at N=2500.
#  - NaNs: with little data you usually IMPUTE (fillna median) rather than drop,
#    to avoid throwing away scarce rows. State your choice.
#       train = train.fillna(train.median(numeric_only=True))
#  - Outlier decision (state your reasoning, either is fine):
#       keep:  "few rows, no domain info — not discarding information"
#       clip:  "capping >3 SD to limit leverage on a small sample"


# %% ==========================================================================
# PHASE 2 — SPLIT  (small holdout; CV does the real work)
# =============================================================================
from sklearn.model_selection import train_test_split

feats = [f"x_{i}" for i in range(5)]
X = train[feats].values
y = train["y"].values

# Option A (used here): keep a small 20% holdout as a final sanity check,
# but make ALL model decisions via 10-fold CV on the training part.
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)
print(X_tr.shape, X_te.shape)   # ~400 train / ~100 test

# Option B (also valid for very small N): skip the holdout entirely and report
# the 10-fold CV score AS your performance estimate, since CV reuses every row.
# If you do this, say so explicitly and never tune against that score.

CV = 10   # more folds than the large-data case (was 5) -> more stable estimate


# %% ==========================================================================
# PHASE 3 — BAKE-OFF  (CV on TRAINING data; favor regularized/simple models)
# =============================================================================
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, HuberRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score

alphas = np.logspace(-3, 3, 25)

# ---- METRIC: match whatever THEY grade on (you'll learn it in the assessment) --
# sklearn maximizes scores, so error metrics come back NEGATIVE -> flip the sign.
#   "neg_mean_squared_error"        -> MSE   LOWER better
#   "neg_root_mean_squared_error"   -> RMSE  LOWER better
#   "neg_mean_absolute_error"       -> MAE   LOWER better (robust to outliers)
#   "r2"                            -> R^2   HIGHER better (remove the minus below)
SCORING = "neg_mean_squared_error"   # <-- change to match their metric

models = {
    "OLS":        make_pipeline(StandardScaler(), LinearRegression()),
    "Ridge":      make_pipeline(StandardScaler(), RidgeCV(alphas=alphas)),
    "Lasso":      make_pipeline(StandardScaler(), LassoCV(cv=CV, max_iter=5000)),
    # Huber = outlier-robust linear regression. Especially useful on small data
    # where a single extreme point has more leverage. Downweights big residuals
    # instead of deleting rows.
    "Huber":      make_pipeline(StandardScaler(), HuberRegressor(max_iter=2000)),
    "Ridge+poly": make_pipeline(StandardScaler(), PolynomialFeatures(2), RidgeCV(alphas=alphas)),
    # Tree models kept for comparison but EXPECT them to overfit small data.
    # Shallower/smaller forests resist overfitting a bit better:
    "RF":         RandomForestRegressor(n_estimators=300, max_depth=4, random_state=0),
    "GBM":        GradientBoostingRegressor(n_estimators=200, max_depth=2, random_state=0),
}

for name, m in models.items():
    # (If SCORING="r2", remove the leading minus.)
    score = -cross_val_score(m, X_tr, y_tr, cv=CV, scoring=SCORING)
    print(f"{name:12s} {score.mean():8.3f} +/- {score.std():.3f}   ({SCORING})")

# SMALL-DATA READING OF THE SCOREBOARD:
#  - The +/- std will be LARGER than with big data. Treat two models as tied
#    unless the gap clearly exceeds the std. Don't over-trust tiny edges.
#  - If a regularized linear model ties or beats RF/GBM, PREFER it: simpler,
#    less overfit-prone, and explainable. On 500 rows this is common.
#  - Ridge+poly still catches interactions; degree=2 only.


# %% ==========================================================================
# PHASE 3b — OVERFIT CHECK (especially important on small data)
# Compare in-sample (train) MSE vs CV MSE. A big gap = the model is memorizing.
# =============================================================================
from sklearn.metrics import mean_squared_error

for name, m in models.items():
    m.fit(X_tr, y_tr)
    train_mse = mean_squared_error(y_tr, m.predict(X_tr))
    cv_mse = -cross_val_score(m, X_tr, y_tr, cv=CV,
                              scoring="neg_mean_squared_error").mean()
    gap = cv_mse - train_mse
    print(f"{name:12s} train {train_mse:8.3f} | CV {cv_mse:8.3f} | gap {gap:8.3f}")

# A near-zero train MSE with a much higher CV MSE (large gap) = overfitting.
# Tree models often show this on small data. The model with the LOWEST CV MSE
# AND a modest gap is your best, most trustworthy choice.


# %% ==========================================================================
# PHASE 4 — DIAGNOSE THE WINNER (understand WHY it won)
# =============================================================================
poly = PolynomialFeatures(2, include_bias=False)
Xp = poly.fit_transform(X_tr)
names = poly.get_feature_names_out(feats)

r = RidgeCV(alphas=alphas).fit(StandardScaler().fit_transform(Xp), y_tr)
order = np.argsort(np.abs(r.coef_))[::-1]
print("Top terms by |coefficient|:")
for i in order[:8]:
    print(f"  {names[i]:12s} {r.coef_[i]:8.3f}")

# Large coeff on a product term (e.g. "x_1 x_3") = hidden interaction.
# Write the discovered structure as a markdown cell.


# %% ==========================================================================
# PHASE 5 — FINALIZE
# =============================================================================
from sklearn.metrics import mean_squared_error, r2_score

# Retrain the chosen pipeline on ALL available data (every scarce row counts).
final = make_pipeline(StandardScaler(), PolynomialFeatures(2), RidgeCV(alphas=alphas))
final.fit(X, y)

# Sanity check on the small holdout (treat as a rough check, not gospel —
# ~100 points gives a noisy estimate; your CV score is the more reliable number).
pred_te = final.predict(X_te)
print(f"Holdout (noisy) MSE {mean_squared_error(y_te, pred_te):.3f}  "
      f"R2 {r2_score(y_te, pred_te):.3f}")

# If predicting a separate test file:
# test = pd.read_csv("data/test.csv")
# predictions = final.predict(test[feats].values)

# FINAL MARKDOWN SUMMARY (for the grader) — emphasize the small-data reasoning:
#   "N~500, so I used 10-fold CV as the primary metric (holdout too small to
#    trust alone). Preferred regularized linear models to limit overfitting;
#    checked train-vs-CV gap to confirm. Signal ~= ...; selected Ridge+poly
#    (degree 2) with CV-MSE X. With more data I'd revisit tree models."
#
# Kernel -> Restart & Run All before submitting.

# =============================================================================
# SMALL-DATA CHEAT-SHEET (the mindset shifts)
# =============================================================================
#   - CV is the truth; a single holdout is too noisy at N=500. Use cv=10.
#   - Prefer simpler/regularized models; they generalize better when data scarce.
#   - Always check train-vs-CV gap; big gap = overfitting (common with trees).
#   - Keep PolynomialFeatures at degree=2 (degree 3 = 55 cols, too many).
#   - Impute rather than drop NaNs; outliers have more leverage — look closely.
#   - Larger +/- std on every score; don't over-interpret small differences.
# =============================================================================

```


