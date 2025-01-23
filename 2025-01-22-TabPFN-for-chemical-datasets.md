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

#ArtificialIntelligence #DrugDiscovery #Cheminformatics #AIForScience #AIInChemistry

---

## TabPFN

A recent paper published in Nature, [Accurate predictions on small data with a tabular foundation model by Hollmann et al.](https://doi.org/10.1038/s41586-024-08328-6), describes a new Deep Learning (DL) algorithm for making predictions on tabular data, TabPFN (Tabular Prior-data Fitted Network). TabPFN is a transformer-based foundation model for tabular data that operates via in-context learning (ICL), enabling it to train and predict on an entire dataset in a single forward pass.

The authors generated over 100 million synthetic datasets using structural causal models (SCMs). These models simulate causal relationships and mimic real-world tabular data challenges, such as non-linear relationships, missing values, outliers, and diverse feature types (e.g., categorical, ordinal, numerical). These datasets sampled high-level hyperparameters (such as dataset size and number of features), both classification and regression tasks, and Gaussian noise was added to mimic real-world complexities.

These synthetic datasets were used to pre-train a transformer model. During the training, parts of the datasets were masked, and the model is trained to predict masked target values in synthetic datasets, given features and labelled samples as context. The parameters of the neural network were updated until the predictions matched the masked values. Through this training, the model learns to fill in missing data from a dataset. The model learns a generic prediction algorithm that approximates Bayesian inference for the synthetic data prior, enabling robust handling of unseen datasets.

When making predictions TabPFN uses ICL and processes an entire dataset (both labelled and test samples) in one pass, performing training and inference simultaneously.

## Therapeutic data commons (TDC)

The [Therapeutic data commons (TDC)](https://tdcommons.ai/) aims to help development of AI/ML tools for therapeutic science by providing datasets and curated benchmarks to assess the performance of new methods. The TDC ADMET benchmark group contains 22 datasets for molecular property prediction, ranging from 475 to 13,130 entries, with both classification and regression tasks. The datasets contain SMILES strings for each chemical compound, and a target variable to be predicted. These can all be easily downloaded using their Python library.

## Using TabPFN on TDC datasets

TabPFN is designed for small tabular datasets, and the default parameters limit to training on 10,000 entries with 500 features. If a dataset had more than 10,000 entries in the training and validation sets, a random selection of 10,000 entries was used. For each dataset, the training and validation data were combined, and the test set put aside. For all entries, the 210 RDKit descriptors were calculated and used as the features for training. Predictions were made on the test set and compared to the true values to evaluate the performance of TabPFN. The training was repeated using molecular fingerprints (MACCS keys and ECPF folded to 500 bits), but the performance was found to be lower than using RDKit calculated properties.

**NOTE**: Due to memory limits on my computer, I had to limit to datasets with fewer than 1,800 entries in the training data. Training models for the larger datasets is ongoing.

## Performance on TDC datasets

Using the RDKit calculated descriptors as features, TabPFN come in the top 10 models for all TDC datasets apart from "Clearance_Hepatocyte_Az". It comes 3rd for the "Vdss_Lombardo" dataset, 2nd for "Caco2_Wang", "Pgp_Broccatelli", and "Bbb_Martins" datasets, and is the highest performing model for the "Clearance_Microsome_Az" dataset. The high performing datasets include both classification and regression tasks and doesn't seem to be a clear link between classification performance and imbalanced datasets. "Cyp2C9_Substrate_Carbonmangels" is somewhat imbalanced (19.3% positive - TDC rank: 10th) but so are "HIA_Hou" (11.1% negative - TDC rank: 5th), and "Bioavailability_Ma", (22.9% negative - TDC rank: 5th). The strongest link seems to be between dataset size and performance, with all the highest performing models having >900 entries.

The training time is an average of five repeats, using WSL on a Windows machine with 8 GB RAM allocated to WSL.

| Dataset | Size | Task | Metric | Training time (min) | TabFPN performance | Current TDC best performance | TabPFN TDC leaderboard rank |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Caco2_Wang | 906 | Regression | MAE | 8.25 | 0.282 ± 0.005 | 0.276 ± 0.005 | 2nd |
| HIA_Hou | 578 | Classification | AUROC | 3.17 | 0.987 ± 0.001 | 0.990 ± 0.002 | 5th |
| Pgp_Broccatelli | 1218 | Classification | AUROC | 12.77 | 0.936 ± 0.004 | 0.938 ± 0.002 | 2th |
| Bioavailability_Ma | 640 | Classification | AUROC | 4.32 | 0.735 ± 0.016 | 0.753 ± 0.000 | 5th |
| Bbb_Martins | 2030 | Classification | AUROC | X.XX | 0.917 ± 0.003 | 0.920 ± 0.006 | 2nd |
| Vdss_Lombardo | 1130 | Regression | Spearman | 12.91 | 0.693 ± 0.004 | 0.713 ± 0.007 | 3rd |
| Cyp2D6_Substrate_Carbonmangels | 667 | Classification | AUPRC | 4.82 | 0.714 ± 0.009 | 0.736 | 6th |
| Cyp3A4_Substrate_Carbonmangels | 670 | Classification | AUROC | 3.97 | 0.641 ± 0.004 | 0.667 ± 0.019 | 7th |
| Cyp2C9_Substrate_Carbonmangels | 669 | Classification | AUPRC | 4.28 | 0.400 ± 0.013 | 0.441 ± 0.033 | 10th |
| Half_Life_Obach | 667 | Regression | Spearman | 4.20 | 0.546 ± 0.013 | 0.576 ± 0.025 | 6th |
| Clearance_Microsome_Az | 1102 | Regression | Spearman | 12.71 | 0.632 ± 0.006 | 0.630 ± 0.010 | 1st |
| Clearance_Hepatocyte_Az | 1213 | Regression | Spearman | 11.27 | 0.391 ± 0.004 | 0.536 ± 0.02 | >10th |
| Herg | 655 | Classification | AUROC | 3.54 | 0.850 ± 0.002 | 0.880 ± 0.002 | 6th |
| Dili | 475 | Classification | AUROC | 1.92 | 0.910 ± 0.005 | 0.925 ± 0.005 | 6th |

## Advantages

On some of the TDC datasets, TabPFN exhibits state-of-the-art performance out of the box. Regularly outperforming other DL methods such as graph neural networks (GNNs) designed specifically for cheminformatics. It can learn complex relationships between features and the target on small datasets due to the pre-training on synthetic data. TabPFN uses a scikit-learn style API, making it very easy to use and integrate into existing workflows. Since TabPFN approximates Bayesian inference, it can include important features such as uncertainty estimation possible.

## Limitations

TabPFN is limited to 10,000 entries with 500 features as a default, as this is the limit of the synthetic data it was trained on. It is possible to use datasets larger than this using the `ignore_pretraining_limits=True` parameter, but this may lead to very long training times. The computational requirements for TabPFN scale quadratically with the number of samples (n) and the number of features (m), i.e. O(n<sup>2</sup> + m<sup>2</sup>) so training and predicting on larger datasets gets increasing longer. The training time can be long, though not significantly worse than other DL methods such as Chemprop in my experience. The single pass training and prediction architecture is not great for situations when you need to make multiple predictions, but the authors have included an option to cache the effect of the training data (using the fit_mode="fit_with_cache" parameter) when training the model, making the training and prediction API much more like a standard scikit-learn ML model. 

## Conclusions

TabPFN is definitely a method to consider when building QSAR models, especially when you have a small dataset. With more time (and computing power), I'd be interested to see if combining the RDKit calculated descriptors with a molecular fingerprint (i.e. adding MACCS keys would make 376 feature columns) improves performance. In my experience sometimes combining the local description of molecules using molecular fingerprints with global descriptions of molecules using calculated descriptors creates a very good embedding of your molecules.

It's possible to fine tune TabPFN with specialised datasets. Perhaps fine tuning with a large number of chemical datasets, or synthetic datasets of calculated molecular properties would create a chemical tabular foundation model (ChemTabPFN?), capable of accurately predicting chemical property-structure relationships.

## References

- The code used can be found in [this Github repo](https://github.com/jonswain/tabpfn-tdc)
- [TabPFN Github repo](https://github.com/PriorLabs/tabpfn)
