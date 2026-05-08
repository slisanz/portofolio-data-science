"""Refactor 01_eda.ipynb so it consumes transactions_clean.parquet
and contains pure EDA only (no cleaning logic)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell("""# 01 Exploratory Data Analysis

Pure EDA on the cleaned transactions. The cleaning step lives in `00_data_cleaning.ipynb`; this notebook reads the artifact `transactions_clean.parquet` and assumes zero NaN, parsed dates, and internally consistent monetary columns. The goal here is to understand what the data says, not to fix it.

Roadmap:

1. Load the cleaned table and confirm it is clean.
2. Descriptive statistics, split by dtype.
3. Revenue by branch, by product line, by customer type, by payment method.
4. Monthly trend.
5. Rating distribution by branch.
6. Findings that the rest of the pipeline acts on.
"""))

cells.append(nbf.v4.new_code_cell("""import os, warnings
os.environ.setdefault('LOKY_MAX_CPU_COUNT', '4')
warnings.filterwarnings('ignore')

import sys
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src import config
from src.viz import set_style, fmt_money

set_style()
RNG = np.random.default_rng(config.RANDOM_SEED)
"""))

cells.append(nbf.v4.new_markdown_cell("## 1. Load the cleaned table"))

cells.append(nbf.v4.new_code_cell("""df = pd.read_parquet(config.DATA_PROCESSED / 'transactions_clean.parquet')
print('shape:', df.shape)
assert df.isna().sum().sum() == 0, 'cleaned table contains NaN, re-run notebook 00'
print('NaN total:', int(df.isna().sum().sum()))
df.head()
"""))

cells.append(nbf.v4.new_markdown_cell("""## 2. Descriptive statistics

Split by dtype so the output stays readable. `describe(include='all')` would merge both kinds in a single table and pad the cells where a statistic does not apply (mean of a string, unique count of a number) with NaN. Those NaNs are formatting placeholders, not real missing values. The cleaning step in notebook 00 already guarantees zero true NaN.
"""))

cells.append(nbf.v4.new_code_cell("""num_cols = df.select_dtypes(include='number').columns
cat_cols = ['Branch', 'City', 'Customer type', 'Gender', 'Product line', 'Payment']

print('Numeric features')
display(df[num_cols].describe().T.round(3))

print()
print('Categorical features')
display(df[cat_cols].describe().T)
"""))

cells.append(nbf.v4.new_markdown_cell("## 3. Revenue by branch"))

cells.append(nbf.v4.new_code_cell("""rev_by_branch = df.groupby('Branch')['Total'].sum().sort_values(ascending=False)
print(rev_by_branch.apply(fmt_money))
ax = rev_by_branch.plot(kind='bar', color=['#1f6feb', '#10b981', '#f59e0b'])
ax.set_title('Total Revenue by Branch')
ax.set_ylabel('Revenue (USD)')
plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("## 4. Revenue by product line"))

cells.append(nbf.v4.new_code_cell("""rev_by_pl = df.groupby('Product line')['Total'].sum().sort_values(ascending=True)
print(rev_by_pl.apply(fmt_money))
ax = rev_by_pl.plot(kind='barh', color='#1f6feb')
ax.set_title('Revenue by Product Line')
ax.set_xlabel('Revenue (USD)')
plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("## 5. Customer type contribution"))

cells.append(nbf.v4.new_code_cell("""ct = df.groupby('Customer type').agg(
    Revenue=('Total', 'sum'),
    Transactions=('Invoice ID', 'count'),
    AvgBasket=('Total', 'mean'),
    AvgRating=('Rating', 'mean'),
).round(2)
ct
"""))

cells.append(nbf.v4.new_markdown_cell("## 6. Payment mix"))

cells.append(nbf.v4.new_code_cell("""pay = df.groupby('Payment').agg(
    Revenue=('Total', 'sum'),
    Transactions=('Invoice ID', 'count'),
).round(2)
pay['RevenueShare'] = (pay['Revenue'] / pay['Revenue'].sum() * 100).round(1)
pay.sort_values('Revenue', ascending=False)
"""))

cells.append(nbf.v4.new_markdown_cell("## 7. Monthly trend"))

cells.append(nbf.v4.new_code_cell("""monthly = df.assign(Month=df['Date'].dt.to_period('M').astype(str)).groupby('Month')['Total'].sum()
print(monthly.apply(fmt_money))
ax = monthly.plot(marker='o', color='#1f6feb')
ax.set_title('Monthly Revenue')
ax.set_ylabel('Revenue (USD)')
plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("## 8. Rating distribution by branch"))

cells.append(nbf.v4.new_code_cell("""ax = sns.boxplot(data=df, x='Branch', y='Rating', hue='Branch',
                 palette=['#1f6feb', '#10b981', '#f59e0b'], legend=False)
ax.set_title('Customer Rating by Branch')
ax.set_ylabel('Rating')
plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("""## Findings

1. The three branches are within a few percent of each other on revenue. Branch leadership shifts depending on the metric, so business actions should be branch specific rather than blanket.
2. Fashion accessories and Food and beverages dominate the revenue ranking, while Food and beverages also leads on customer rating. Worth shelf placement priority and bundle promotion.
3. Members contribute a sizeable share of revenue and have a higher average basket than Normal customers, which justifies the segmentation work in notebook 03.
4. Cash and Ewallet are nearly tied on revenue, signalling room to migrate volume into digital rails through promotions on the Ewallet side.
5. The dataset spans January to March 2019. Any monthly story is short: this notebook reports the trend for context but the forecasting in notebook 04 keeps the horizon to 30 days for that reason.
"""))

nb['cells'] = cells
nb['metadata'] = {
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python'},
}
nbf.write(nb, 'notebooks/01_eda.ipynb')
print('01_eda.ipynb refactored')
