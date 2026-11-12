# Weather-Based Outfit Recommendation - EDA (Phases 1-5)
# In VS Code: click "Run Cell" above each "# %%" to run step by step,
# or just run the whole file:  python eda.py
# %%
import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

matplotlib.use("Agg") if os.environ.get("NO_SHOW") else None
sns.set_theme(style="whitegrid")
os.makedirs("plots", exist_ok=True)

def save(name):
    plt.tight_layout()
    plt.savefig(f"plots/{name}.png", dpi=120)
    if not os.environ.get("NO_SHOW"):
        plt.show()
    plt.close()

df = pd.read_csv("data/weather_outfit.csv")
NUM = ["temperature_c", "rainfall_mm", "humidity_pct", "wind_speed_kmh"]
TARGET = "outfit_category"

# ============ PHASE 1: BASIC DATA UNDERSTANDING ============
# %% 1. Dataset overview
print(df.head(10))
# %% 2. Dataset shape
print("Shape:", df.shape)
# %% 3. Dataset information
df.info()
# %% 4. Statistical summary
print(df.describe(include="all"))
# %% 5. Missing values
print(df.isnull().sum())
# %% 6. Duplicates
print("Duplicate rows:", df.duplicated().sum())

# ============ PHASE 2: UNIVARIATE ANALYSIS ============
# %% 7. Target distribution
plt.figure(figsize=(7, 4))
ax = sns.countplot(data=df, x=TARGET, order=df[TARGET].value_counts().index)
ax.bar_label(ax.containers[0])
plt.title("Outfit Category Distribution")
save("07_target_distribution")
# %% 8. Occasion distribution
plt.figure(figsize=(7, 4))
ax = sns.countplot(data=df, x="occasion", order=df["occasion"].value_counts().index)
ax.bar_label(ax.containers[0])
plt.title("Occasion Distribution")
save("08_occasion_distribution")
# %% 9-12. Numeric distributions
for i, col in enumerate(NUM, start=9):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.histplot(df[col], kde=True, ax=axes[0])
    sns.boxplot(x=df[col], ax=axes[1])
    fig.suptitle(f"{col} Distribution")
    save(f"{i:02d}_{col}_distribution")

# ============ PHASE 3: OUTLIERS & CORRELATION ============
# %% 13. Outlier detection (IQR method)
for col in NUM:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n = ((df[col] < low) | (df[col] > high)).sum()
    print(f"{col:16s} bounds=({low:.1f}, {high:.1f})  outliers={n}")
plt.figure(figsize=(10, 5))
sns.boxplot(data=(df[NUM] - df[NUM].mean()) / df[NUM].std())
plt.title("Outlier Detection (standardised values)")
save("13_outliers")
# %% 14. Correlation heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(df[NUM].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
save("14_correlation_heatmap")

# ============ PHASE 4: BIVARIATE ANALYSIS ============
# %% 15-18. Numeric feature vs outfit category
order = df[TARGET].value_counts().index
for i, col in enumerate(NUM, start=15):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.boxplot(data=df, x=TARGET, y=col, order=order, ax=axes[0])
    sns.violinplot(data=df, x=TARGET, y=col, order=order, ax=axes[1])
    for a in axes:
        a.tick_params(axis="x", rotation=20)
    fig.suptitle(f"{col} vs Outfit Category")
    save(f"{i}_{col}_vs_outfit")
print(df.groupby(TARGET)[NUM].mean().round(2))
# %% 19. Occasion vs outfit category
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="occasion", hue=TARGET)
plt.title("Occasion vs Outfit Category")
save("19_occasion_vs_outfit")
print(pd.crosstab(df["occasion"], df[TARGET], normalize="index").round(2))

# ============ PHASE 5: MULTIVARIATE ANALYSIS ============
# %% 20. Pair plot
g = sns.pairplot(df, vars=NUM, hue=TARGET, corner=True, plot_kws={"alpha": 0.6, "s": 20})
g.figure.suptitle("Pair Plot", y=1.02)
g.figure.savefig("plots/20_pairplot.png", dpi=110)
if not os.environ.get("NO_SHOW"):
    plt.show()
