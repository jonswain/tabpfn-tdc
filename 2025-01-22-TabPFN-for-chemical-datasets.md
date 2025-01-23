---
layout: post
title:  "TabPFN for chemical datasets"
date:   2025-01-22 12:00:00 +0100
categories: 
    - AI
    - cheminformatics 
    - data science 
    - machine learning
---

Deep Learning models have traditionally performed well on unstructured data such as text and images, but poorly on structured tabular data, and are usually outperformed by Gradient Boosted Decision Trees (GBDTs) on tabular chemical data. TabPFN (Tabular Prior-data Fitted Network) is a transformer-based foundation model for tabular data, pre-trained on millions of synthetic datasets to solve supervised learning tasks, with state-of-the-art performance on benchmarks. But does it work for cheminformatics?

---

## TabPFN

A recent paper published in Nature, “Accurate predictions on small data with a tabular foundation model” by Hollmann et al. (https://doi.org/10.1038/s41586-024-08328-6), describes a new Deep Learning (DL) algorithm for making predictions on tabular data, TabPFN (Tabular Prior-data Fitted Network). TabPFN is a transformer-based foundation model for tabular data that operates via in-context learning (ICL), enabling it to train and predict on an entire dataset in a single forward pass.

The authors generated over 100 million synthetic datasets using structural causal models (SCMs). These models simulate causal relationships and mimic real-world tabular data challenges, such as non-linear relationships, missing values, outliers, and diverse feature types (e.g., categorical, ordinal, numerical). These datasets sampled high-level hyperparamters (such as dataset size and number of features), both classification and regression tasks, and Gaussian noise was added to mimic real-world complexities.

These synthetic datasets were used to pre-train a transformer model. During the training, parts of the datasets were masked and the model model is trained to predict masked target values in synthetic datasets, given features and labeled samples as context. The parameters of the neural network were updated until the predictions matched the masked values. Through this training, the model learns to fill in missing data from a dataset. The model learns a generic prediction algorithm that approximates Bayesian inference for the synthetic data prior, enabling robust handling of unseen datasets.

When making predictions TabPFN uses ICL, and processes an entire dataset (both labeled and test samples) in one pass, performing training and inference simultaneously.

## Therapeutic data commons (TDC)

The Therapeutic data commons (TDC) aims to help development of AI/ML tools for therapeutic science by providing datasets and curated benchmarks to assess the performance of new methods. The TDC ADMET benchmark group contains 22 datasets for molecular property prediction, ranging from 475 to 13,130 entries, with both classification and regression tasks. The datasets contain SMILES strings for each chemical compound, and a target variable to be predicted. These can all be easily downloaded using their Python libarry.

## Using TabPFN on TDC datasets

TabPFN is designed for small tabular datasets, and the default parameters limit to training on 10,000 entries with 500 features. If a dataset had more than 10,000 entries in the training and validation sets, a random selection of 10,000 entries was used. For each dataset, the training and validation data were combined, and the test set put aside. For all entries, the 210 RDKit descriptors were calculated and used as the features for training. Predictions were made on the test set and compared to the true values to evaluate the performance of TabPFN. The training was repeated using molecular fingerprints (MACCS keys and ECPF folded to 500 bits), but the performance was found to be lower than using RDKit calculated properties.

## Performance on TDC datasets

TODO: Look at performance with dataset type, size, and class imbalance
TODO: For training time specify computational resources used

| Dataset | Size | Task | Metric | Training time (min) | TabFPN performance | Current TDC best performance | TabPFN TDC leaderboard rank |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Caco2_Wang | 906 | Regression | MAE | 8.25 | 0.282 ± 0.005 | 0.276 ± 0.005 | 2nd |
| HIA_Hou | 578 | Classification | AUROC | 3.17 | 0.987 ± 0.001 | 0.990 ± 0.002 | 5th |
| Pgp_Broccatelli | 1218 | Classification | AUROC | 12.77 | 0.936 ± 0.004 | 0.938 ± 0.002 | 2th |
| Bioavailability_Ma | 640 | Classification | AUROC | 4.32 | 0.735 ± 0.016 | 0.753 ± 0.000 | 5th |
| Bbb_Martins | 2030 | Classification | AUROC | X.XX | 0.917 ± 0.003 | 0.920 ± 0.006 | 2nd |
| Vdss_Lombardo | 1130 | Regression | Spearman | X.XX | X.XXX ± X.XXX | 0.713 ± 0.007 | nth |
| Cyp2D6_Substrate_Carbonmangels | 667 | Classification | AUPRC | 4.82 | 0.714 ± 0.009 | 0.736 | 6th |
| Cyp3A4_Substrate_Carbonmangels | 670 | Classification | AUROC | 3.97 | 0.641 ± 0.004 | 0.667 ± 0.019 | 7th |
| Cyp2C9_Substrate_Carbonmangels | 669 | Classification | AUPRC | 4.28 | 0.400 ± 0.013 | 0.441 ± 0.033 | 10th |
| Half_Life_Obach | 667 | Regression | Spearman | X.XX | X.XXX ± X.XXX | 0.576 ± 0.025 | nth |
| Clearance_Microsome_Az | 1102 | Regression | Spearman | 12.71 | 0.632 ± 0.006 | 0.630 ± 0.010 | 1st |
| Clearance_Hepatocyte_Az | 1213 | Regression | Spearman | 11.27 | 0.391 ± 0.004 | 0.536 ± 0.02 | >10th |
| Herg | 655 | Classification | AUROC | 3.54 | 0.850 ± 0.002 | 0.880 ± 0.002 | 6th |
| Dili | 475 | Classification | AUROC | 1.92 | 0.910 ± 0.005 | 0.925 ± 0.005 | 6th |

## Advantages

- Smaller tabular datasets where DL using graph neural networks (GNNs) doesn't perform well.
- Uses scikit-learn style API, very easy to use.
- Approximates Bayesian inference, making uncertainy estimation possible.

## Limitations

- Limited to 10,000 entries with 500 features.
- Slow training
- The computational requirements for TabPFN scale quadratically with the number of samples (n) and the number of features (m), i.e. O(n<sup>2</sup> + m<sup>2</sup>). 
- Single pass training and prediction not useful for multiple predictions. The authors have included an option to cache the effect of the training data (using the fit_mode="fit_with_cache" parameter) when training the model, making the training and prediction API much more like a standard scikit-learn ML model. 

## Conclusions

- Is it possible to fine tune on chemical datasets, or create synthetic datasets that resemble common relationships in cheminformatics?
- Extensions - maybe combine MACCS key with RDKit calculated descriptors
