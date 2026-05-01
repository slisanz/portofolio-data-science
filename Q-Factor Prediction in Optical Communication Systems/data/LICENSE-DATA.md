# Dataset License — CC BY 4.0

The file `synthetic_qfactor_dataset.csv` in this directory is distributed
under the **Creative Commons Attribution 4.0 International** licence
(<https://creativecommons.org/licenses/by/4.0/>).

## Required Attribution

> Al-Dulaimi, Ahmed; Abdulla, Essam N. (2025). *Large-Scale Synthetic Dataset
> for Q-Factor Prediction in Optical Communication Systems*, V1, Mendeley Data.
> DOI: [10.17632/6fcnwdjxt5.1](https://doi.org/10.17632/6fcnwdjxt5.1).
> Licensed under CC BY 4.0.

## Modifications Made in This Repository

- The raw CSV is preserved unchanged.
- For modeling, a 70/15/15 train/validation/test split is generated and stored
  under `data/processed/` (gitignored).
- Standard scaling is fit on the training split.
- Engineered features (ratios, products, log transforms) are computed at run time.

## Conditions Reminder

- You may share, copy, modify, and redistribute the dataset for any purpose,
  including commercially, **provided you give appropriate credit, link to the
  licence, and indicate if changes were made**.
- You may not suggest the rights holder endorses you or your use.
- Any third-party content embedded within the dataset may require additional
  permission from its respective rights holder.
