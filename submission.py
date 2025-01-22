from time import time

import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors
from tabpfn import TabPFNClassifier, TabPFNRegressor
from tdc.benchmark_group import admet_group
from tqdm import tqdm


def calculateDescriptors(mol: Chem.Mol, missingVal: float | None = None) -> dict:
    """Calculate the full list of descriptors for a molecule."""
    res = {}
    for nm, fn in Descriptors._descList:
        try:
            val = fn(mol)
        except:
            val = missingVal
        res[nm] = val
    return res


def createDescriptorDataFrame(smiles: list[str]) -> pd.DataFrame:
    """Create a DataFrame of descriptors for a list of SMILES strings."""
    mols = [Chem.MolFromSmiles(smi) for smi in smiles]
    descs = [calculateDescriptors(mol) for mol in mols]
    return pd.DataFrame(descs)


def getDeviceType() -> str:
    """Get the device type to use for training and inference."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def main():
    """Train and evaluate the model on the benchmark datasets."""
    # Get the device type to use for training and inference
    device_type = getDeviceType()
    print(f"Training and inference completed using: {device_type}")

    # Load the benchmark datasets
    group = admet_group(path="data/")
    predictions_list = [{}, {}, {}, {}, {}]

    for dataset_name in group.dataset_names:
        print(f"Dataset: {dataset_name}")
        start = time()
        benchmark = group.get(dataset_name)
        train_val, test = benchmark["train_val"], benchmark["test"]

        X_train = createDescriptorDataFrame(train_val["Drug"]).fillna(0)
        y_train = train_val["Y"].fillna(0)
        X_test = createDescriptorDataFrame(test["Drug"]).fillna(0)
        y_test = test["Y"].fillna(0)

        print(f"Train: {X_train.shape}, {y_train.shape}")
        print(f"Test: {X_test.shape}, {y_test.shape}")

        try:
            for seed in tqdm([1, 2, 3, 4, 5]):
                params = {
                    "random_state": seed,
                    "n_jobs": -1,
                    "n_estimators": 4,
                    "device": device_type,
                }
                # TODO: Work out how to use memory saving with MPS
                if device_type == "mps":
                    params["memory_saving_mode"] = False

                # TabPFN has a maximum number of samples it can handle
                if len(X_train) > 10000:
                    X_train = X_train.sample(10000)
                    y_train = y_train.loc[X_train.index]

                if y_test.nunique() == 2:
                    model = TabPFNClassifier(**params)
                    model.fit(X_train, y_train)
                    y_pred = model.predict_proba(X_test)[:, 1]
                else:
                    model = TabPFNRegressor(**params)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                predictions_list[seed - 1][dataset_name] = y_pred
            ave, std = group.evaluate_many(predictions_list)[dataset_name]
            print(f"Performance: {ave:.3f} +/- {std:.3f}")
            end = time()
            print(f"Average time taken: {(end - start)/5:.2f} s")
        except Exception as e:
            print(f"Error: {e}")

    performance = group.evaluate_many(predictions_list, save_file_name="submission.txt")
    print(performance)


if __name__ == "__main__":
    main()
