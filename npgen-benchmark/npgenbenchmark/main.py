import gzip
import itertools
import json
import os
import pickle
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import torch
from fcd_torch import FCD
from npgenbenchmark.classifier import ConvertedModel
from npgenbenchmark.dataset import InferenceDataset
from npgenbenchmark.utils import download
from rdkit import Chem, RDLogger
from scipy.special import rel_entr
from scipy.stats import entropy, gaussian_kde
from torch.utils.data import DataLoader
from tqdm import tqdm

RDLogger.DisableLog("rdApp.*")

PROJECT_ROOT = Path(__file__).parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
INDEX_FILE = PROJECT_ROOT / "index_v1.json"
TRAIN_RESULTS_FILE = PROJECT_ROOT / "data/train_results.csv"
TEST_RESULTS_FILE = PROJECT_ROOT / "data/test_results.csv"

PATHWAY_MODEL = ConvertedModel(7)
SUPERCLASS_MODEL = ConvertedModel(77)
CLASS_MODEL = ConvertedModel(687)


class NPGenBenchmark:
    """
    Runs chemical structure classification using pre-trained NPClassifier models.

    Predicts Pathway, Superclass, and Class for a given list of SMILES strings.
    """

    def __init__(
        self,
        device: str = "cuda:0",
        n_jobs: int = 4,
        model_dir: Union[str, Path] = MODEL_DIR,
        index_file: Union[str, Path] = INDEX_FILE,
        test_results_file: Union[str, Path] = TEST_RESULTS_FILE,
        train_results_file: Union[str, Path] = TRAIN_RESULTS_FILE,
        batch_size: int = 128,  # Added batch_size parameter
        verbose: bool = False,
        n_eval_data: int = 30000,
    ):
        """
        Initializes the NPGenBenchmark instance.

        Args:
            device (str): Device to run inference on (e.g., 'cuda:0', 'cpu').
            n_jobs (int): Number of workers for DataLoader.
            model_dir (Union[str, Path]): Directory containing the pre-trained model files
                                         ('NP_classifier_pathway_V1.pt', etc.).
            index_path (Union[str, Path]): Path to the index JSON file ('index_v1.json').
            batch_size (int): Batch size for inference DataLoader.
        """
        self.device = self._setup_device(device)
        self.n_jobs = n_jobs
        self.batch_size = batch_size
        self.model_dir = Path(model_dir)
        self.index_file = Path(index_file)
        self.test_results_fn = Path(test_results_file)
        self.train_results_fn = Path(train_results_file)
        self.verbose = verbose
        self.n_eval_data = n_eval_data

        self._log(f"Using device: {self.device}")
        self._log(f"Using DataLoader workers: {self.n_jobs}")
        self._log(f"Using batch size: {self.batch_size}")

        self._load_index()
        self._load_models()

    def _log(self, message: str):
        """Prints message only if verbose is True."""
        if self.verbose:
            print(message, flush=True)

    def _setup_device(self, device_str: str) -> torch.device:
        """Sets up the torch device."""
        if device_str.startswith("cuda") and not torch.cuda.is_available():
            self._log(
                f"Warning: CUDA device '{device_str}' requested but CUDA not available. Using CPU."
            )
            return torch.device("cpu")
        try:
            return torch.device(device_str)
        except Exception as e:
            self._log(
                f"Warning: Invalid device string '{device_str}'. Using CPU. Error: {e}"
            )
            return torch.device("cpu")

    def _load_index(self):
        """Loads and prepares the index data from the JSON file."""
        self._log(f"Loading index from: {self.index_file}")
        if not self.index_file.exists():
            download(self.index_file.name)
        try:
            with open(self.index_file, "r") as r:
                self.index = json.load(r)
            self.index_class = list(self.index["Class"].keys())
            self.index_superclass = list(self.index["Superclass"].keys())
            self.index_pathway = list(self.index["Pathway"].keys())
            self._log("Index loaded successfully.")
        except Exception as e:
            raise IOError(f"Error loading or parsing index JSON file: {e}")

    def _read_np_scorer(self, fn):
        with gzip.open(fn, "rb") as f:
            fscore = pickle.load(f)
        return fscore

    def _load_models(self):
        """Loads the pre-trained PyTorch models."""
        model_files = {
            "pathway": "NP_classifier_pathway_V1.pt",
            "superclass": "NP_classifier_superclass_V1.pt",
            "class": "NP_classifier_class_V1.pt",
        }
        self.models = {
            "pathway": PATHWAY_MODEL,
            "superclass": SUPERCLASS_MODEL,
            "class": CLASS_MODEL,
        }

        for file in model_files.values():
            model_path = self.model_dir / file
            if not model_path.exists():
                download(f"{self.model_dir.name}/{file}")

        for name, filename in model_files.items():
            model_path = self.model_dir / filename
            self._log(f"Loading {name} model from: {model_path}")
            if not model_path.is_file():
                raise FileNotFoundError(
                    f"{name.capitalize()} model file not found at: {model_path}"
                )

            try:
                state_dict = torch.load(model_path)
                self.models[name].load_state_dict(state_dict)
                self.models[name].eval()
                self._log(
                    f"{name.capitalize()} model loaded successfully to {self.device}."
                )
            except Exception as e:
                raise IOError(f"Error loading {name} model '{model_path}': {e}")

        self.pathway_model = self.models["pathway"].to(self.device)
        self.superclass_model = self.models["superclass"].to(self.device)
        self.class_model = self.models["class"].to(self.device)

        self.np_scorer = self._read_np_scorer(
            os.path.join(self.model_dir, "publicnp.model.gz")
        )

    def _analyze(
        self, pred_path: np.ndarray, pred_super: np.ndarray, pred_class: np.ndarray
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Analyzes raw model predictions to determine classifications based on hierarchy.
        This is the refactored 'analyze' function using class attributes for index data.

        Args:
            pred_path (np.ndarray): Pathway prediction probabilities.
            pred_super (np.ndarray): Superclass prediction probabilities.
            pred_class (np.ndarray): Class prediction probabilities.

        Returns:
            tuple[List[str], List[str], List[str]]: Lists of pathway, superclass,
                                                     and class names.
        """
        threshold = 0.5  # Using the threshold from the original script
        n_path = np.where(pred_path >= threshold)[0].tolist()
        n_super = np.where(pred_super >= threshold)[0].tolist()
        n_class = np.where(pred_class >= threshold)[0].tolist()

        class_result = []
        superclass_result = []
        pathway_result = []

        path_from_class = []
        path_from_superclass = []

        for i in n_class:
            path_from_class = (
                path_from_class + self.index["Class_hierarchy"][str(i)]["Pathway"]
            )
        for j in n_super:
            path_from_superclass = (
                path_from_superclass + self.index["Super_hierarchy"][str(j)]["Pathway"]
            )

        path_from_class = list(set(path_from_class))
        path_from_superclass = list(set(path_from_superclass))

        path_for_vote = n_path + path_from_class + path_from_superclass
        path = list(set([k for k in path_for_vote if path_for_vote.count(k) == 3]))

        if path == []:
            path = list(set([k for k in path_for_vote if path_for_vote.count(k) == 2]))
            if len(path) > 1:
                path = list(
                    set([k for k in path_for_vote if path_for_vote.count(k) == 2])
                )
        if path == []:
            for w in n_path:
                pathway_result.append(self.index_pathway[w])
            if pathway_result == []:
                pathway_result = ["Unclassified"]
            if superclass_result == []:
                superclass_result = ["Unclassified"]
            if class_result == []:
                class_result = ["Unclassified"]
            return pathway_result, superclass_result, class_result

        else:  # path != []
            if set(n_path) & set(path) != set():
                if set(path) & set(path_from_superclass) != set():
                    n_super = [
                        l
                        for l in n_super
                        if set(path)
                        & set(self.index["Super_hierarchy"][str(l)]["Pathway"])
                        != set()
                    ]
                    if n_super == []:
                        n_class = [
                            m
                            for m in n_class
                            if set(path)
                            & set(self.index["Class_hierarchy"][str(m)]["Pathway"])
                            != set()
                        ]
                        n_super = [
                            self.index["Class_hierarchy"][str(n)]["Superclass"]
                            for n in n_class
                        ]
                        n_super = list(set(itertools.chain.from_iterable(n_super)))

                    elif len(n_super) > 1:  # super != []
                        n_class = [
                            u
                            for u in n_class
                            if set(path)
                            & set(self.index["Class_hierarchy"][str(u)]["Pathway"])
                            != set()
                        ]
                        if n_class != []:
                            n_super = [
                                self.index["Class_hierarchy"][str(v)]["Superclass"]
                                for v in n_class
                            ]
                            n_path = [
                                self.index["Class_hierarchy"][str(v)]["Pathway"]
                                for v in n_class
                            ]
                            n_path = list(set(itertools.chain.from_iterable(n_path)))
                            n_super = list(set(itertools.chain.from_iterable(n_super)))

                        elif len(path) == 1:
                            n_super = [np.argmax(pred_super)]
                            n_class = [
                                m
                                for m in [np.argmax(pred_class)]
                                if set(n_super)
                                & set(
                                    self.index["Class_hierarchy"][str(m)]["Superclass"]
                                )
                                != set()
                            ]

                    else:
                        n_class = [
                            o
                            for o in n_class
                            if set(n_super)
                            & set(self.index["Class_hierarchy"][str(o)]["Superclass"])
                            != set()
                        ]
                        if n_class == []:
                            n_class = [
                                m
                                for m in [np.argmax(pred_class)]
                                if set(n_super)
                                & set(
                                    self.index["Class_hierarchy"][str(m)]["Superclass"]
                                )
                                != set()
                            ]
                else:
                    n_class = [
                        p
                        for p in n_class
                        if set(path)
                        & set(self.index["Class_hierarchy"][str(p)]["Pathway"])
                        != set()
                    ]
                    n_super = [
                        self.index["Class_hierarchy"][str(q)]["Superclass"]
                        for q in n_class
                    ]

                    n_super = list(set(itertools.chain.from_iterable(n_super)))

            else:
                n_super = [
                    l
                    for l in n_super
                    if set(path) & set(self.index["Super_hierarchy"][str(l)]["Pathway"])
                    != set()
                ]
                if n_super == []:
                    n_class = [
                        m
                        for m in n_class
                        if set(path)
                        & set(self.index["Class_hierarchy"][str(m)]["Pathway"])
                        != set()
                    ]
                    n_super = [
                        self.index["Class_hierarchy"][str(n)]["Superclass"]
                        for n in n_class
                    ]
                    n_path = [
                        self.index["Class_hierarchy"][str(v)]["Pathway"]
                        for v in n_class
                    ]
                    n_path = list(set(itertools.chain.from_iterable(n_path)))
                    n_super = list(set(itertools.chain.from_iterable(n_super)))

                elif len(n_super) > 1:  # super != []
                    n_class = [
                        u
                        for u in n_class
                        if set(path)
                        & set(self.index["Class_hierarchy"][str(u)]["Pathway"])
                        != set()
                    ]
                    n_super = [
                        self.index["Class_hierarchy"][str(v)]["Superclass"]
                        for v in n_class
                    ]
                    n_path = [
                        self.index["Class_hierarchy"][str(v)]["Pathway"]
                        for v in n_class
                    ]
                    n_path = list(set(itertools.chain.from_iterable(n_path)))
                    n_super = list(set(itertools.chain.from_iterable(n_super)))

                else:
                    n_class = [
                        o
                        for o in n_class
                        if set(path)
                        & set(self.index["Class_hierarchy"][str(o)]["Pathway"])
                        != set()
                    ]
                    n_super = [
                        self.index["Class_hierarchy"][str(v)]["Superclass"]
                        for v in n_class
                    ]
                    n_path = [
                        self.index["Class_hierarchy"][str(v)]["Pathway"]
                        for v in n_class
                    ]
                    n_path = list(set(itertools.chain.from_iterable(n_path)))
                    n_super = list(set(itertools.chain.from_iterable(n_super)))

        for r in path:
            pathway_result.append(self.index_pathway[r])
        for s in n_super:
            superclass_result.append(self.index_superclass[s])
        for t in n_class:
            class_result.append(self.index_class[t])

        if pathway_result == []:
            pathway_result = ["Unclassified"]
        if superclass_result == []:
            superclass_result = ["Unclassified"]
        if class_result == []:
            class_result = ["Unclassified"]
        return pathway_result, superclass_result, class_result

    @staticmethod
    def compute_kl_divergence_npscore_kde(ref_df, gen_df):
        ref_scores = ref_df["NP_Score"].dropna().values
        gen_scores = gen_df["NP_Score"].dropna().values
        X_baseline = ref_scores
        X_sampled = gen_scores
        kde_P = gaussian_kde(X_baseline)
        kde_Q = gaussian_kde(X_sampled)
        x_eval = np.linspace(
            np.hstack([X_baseline, X_sampled]).min(),
            np.hstack([X_baseline, X_sampled]).max(),
            num=1000,
        )
        P = kde_P(x_eval) + 1e-10
        Q = kde_Q(x_eval) + 1e-10
        return entropy(P, Q)

    @staticmethod
    def compute_kl_divergence_from_df(ref_df, gen_df, columns):
        results = {}

        for col in columns:
            # 1. Get unique classes from ref_df (optionally excluding 'UNK')
            unique_classes = ref_df[col].dropna().unique()

            # 2. Get value counts (frequencies)
            ref_counts = ref_df[col].value_counts()
            gen_counts = gen_df[col].value_counts()

            # 3. Create aligned probability distributions
            ref_probs = np.array(
                [ref_counts.get(cls, 0) for cls in unique_classes], dtype=float
            )
            gen_probs = np.array(
                [gen_counts.get(cls, 0) for cls in unique_classes], dtype=float
            )

            # 5. Normalize to probability distributions
            ref_probs = ref_probs / ref_probs.sum()
            gen_probs = gen_probs / gen_probs.sum()

            # ref_probs = np.clip(ref_probs, 1e-10, 1.0)
            gen_probs = np.clip(gen_probs, 1e-10, 1.0)

            # 7. Compute KL divergence
            kl_div = np.sum(rel_entr(ref_probs, gen_probs))
            results[col] = kl_div

        return results

    def run_np_classifier(self, smiles_list: List[str]) -> List[Dict]:
        """
        Performs benchmark classification for a list of SMILES strings.

        Args:
            smiles_list (List[str]): A list of SMILES strings to classify.

        Returns:
            List[Dict]: A list of dictionaries, each containing the classification
                        results for one SMILES string.
        """
        if not isinstance(smiles_list, list):
            raise TypeError("Input must be a list of SMILES strings.")
        if not smiles_list:
            return []

        self._log(f"Starting benchmark for {len(smiles_list)} valid SMILES...")
        start_time = time.time()

        # Create dataset and dataloader
        try:
            dataset = InferenceDataset(smiles_list, self.np_scorer)
            loader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.n_jobs,
                pin_memory=self.device.type == "cuda",
            )
        except Exception as e:
            raise RuntimeError(f"Error creating Dataset or DataLoader: {e}")

        pathway_preds = []
        superclass_preds = []
        class_preds = []
        is_glycosides = []
        processed_smiles = []
        np_scores = []

        # Inference loop
        self._log("Running inference...")
        inference_start_time = time.time()
        for batch in tqdm(loader, desc="Inferencing", ncols=80):
            try:
                features_f = batch["fp_f"].to(self.device)
                features_b = batch["fp_b"].to(self.device)
            except Exception as e:
                raise RuntimeError(
                    f"Error moving batch data to device '{self.device}': {e}"
                )

            with torch.no_grad():
                try:
                    pathway_pred = self.pathway_model(features_f, features_b)
                    superclass_pred = self.superclass_model(features_f, features_b)
                    class_pred = self.class_model(features_f, features_b)
                except Exception as e:
                    raise RuntimeError(f"Error during model inference: {e}")

            pathway_preds.append(pathway_pred.cpu())
            superclass_preds.append(superclass_pred.cpu())
            class_preds.append(class_pred.cpu())
            is_glycosides.append(batch["is_glycoside"])
            processed_smiles.extend(batch["smiles"])
            np_scores.extend(batch["npscore"])

        inference_end_time = time.time()
        self._log(
            f"Inference finished in {inference_end_time - inference_start_time:.2f} seconds."
        )

        # Concatenate predictions from all batches
        try:
            pathway_pred_all = torch.cat(pathway_preds, dim=0)
            superclass_pred_all = torch.cat(superclass_preds, dim=0)
            class_pred_all = torch.cat(class_preds, dim=0)
            is_glycosides_all = torch.cat(is_glycosides, dim=0)
        except Exception as e:
            raise RuntimeError(f"Error concatenating batch results: {e}")

        self._log("Analyzing predictions...")
        analysis_start_time = time.time()
        results = []
        # Iterate through each sample's predictions and analyze
        for i in range(len(processed_smiles)):
            _pathway_pred_np = pathway_pred_all[i].numpy()
            _superclass_pred_np = superclass_pred_all[i].numpy()
            _class_pred_np = class_pred_all[i].numpy()
            _is_glycoside = is_glycosides_all[i]
            _smi = processed_smiles[i]
            _np_score = np_scores[i].cpu().item()

            pathway_result, superclass_result, class_result = self._analyze(
                _pathway_pred_np,
                _superclass_pred_np,
                _class_pred_np,
            )

            results.append(
                {
                    "SMILES": _smi,
                    "Class_Results": ", ".join(class_result),
                    "Superclass_Results": ", ".join(superclass_result),
                    "Pathway_Results": ", ".join(pathway_result),
                    "Is_Glycoside": _is_glycoside.item(),  # Get Python bool/int
                    "NP_Score": _np_score,
                }
            )
        analysis_end_time = time.time()
        self._log(
            f"Analysis finished in {analysis_end_time - analysis_start_time:.2f} seconds."
        )

        end_time = time.time()
        self._log(f"Benchmark completed in {end_time - start_time:.2f} seconds.")
        return results

    def run_benchmark(self, smiles_list: List[str]) -> Dict:
        """
        Main method to run the benchmark.

        Args:
            smiles_list (List[str]): A list of SMILES strings to classify.

        Returns:
            Dict: A dictionary, with the benchmark metrics.
        """
        benchmark_results = {}

        # get smiles
        assert len(smiles_list) >= self.n_eval_data, "Number of SMILES must be greater than or equal to the number of evaluation data."
        if len(smiles_list) > self.n_eval_data:
            smiles_list = random.sample(smiles_list, self.n_eval_data)
            self._log(
                f"Sampling {self.n_eval_data} SMILES from {len(smiles_list)} total SMILES."
            )

        '''if len(smiles_list) >= self.n_eval_data:
            # randomly sample
            self._log(
                f"Sampling {self.n_eval_data} SMILES from {len(smiles_list)} total SMILES."
            )
            smiles_list = random.sample(
                smiles_list, min(len(smiles_list), self.n_eval_data)
            )
        else:
            self._log(
                f"Using {len(smiles_list)} SMILES for evaluation (less than {self.n_eval_data})."
            )'''

        # obtain valild smiles from smiles_list
        can_smiles_list = []
        for smi in smiles_list:
            try:
                can_smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi))
                can_smiles_list.append(can_smi)
            except:
                pass
        benchmark_results["validity"] = len(can_smiles_list) / len(smiles_list)

        # uniqueness
        unique_can_smiles_list = list(set(can_smiles_list))
        benchmark_results["uniqueness"] = len(unique_can_smiles_list) / len(can_smiles_list)

        # Get test SMILES from test results file
        test_result_fn = os.path.join(self.model_dir, self.test_results_fn)
        test_df = pd.read_csv(test_result_fn)
        test_smis = test_df["SMILES"].tolist()
        test_smis_set = set(test_smis)

        # Get train SMILES from train results file
        train_result_fn = os.path.join(self.model_dir, self.train_results_fn)
        train_df = pd.read_csv(train_result_fn)
        train_smis = train_df["SMILES"].tolist()
        train_smis_set = set(train_smis)

        # check novelty
        benchmark_results["novelty"] = len(
            set(unique_can_smiles_list) - train_smis_set
        ) / len(unique_can_smiles_list)

        # get unique smiles, this will be evaluated
        to_eval_smiles_list = can_smiles_list

        # run np
        np_classified_data = self.run_np_classifier(to_eval_smiles_list)
        np_classified_df = pd.DataFrame(np_classified_data)

        # compute kl divergence of npscore
        benchmark_results["NP_Score_KLD"] = self.compute_kl_divergence_npscore_kde(
            test_df, np_classified_df
        )

        # compute KL divergence of class, superclass, pathway
        class_kl = self.compute_kl_divergence_from_df(
            test_df,
            np_classified_df,
            ["Pathway_Results", "Superclass_Results", "Class_Results"],
        )
        benchmark_results.update(class_kl)

        # compute FCD
        fcd_computer = FCD(device=self.device, n_jobs=8)
        fcd_val = fcd_computer(to_eval_smiles_list, test_smis)
        benchmark_results["FCD"] = fcd_val

        return benchmark_results


if __name__ == "__main__":
    import pandas as pd

    input_file = PROJECT_ROOT / "test/coconut_csv_lite-02-2025.csv"
    df = pd.read_csv(input_file, low_memory=False, nrows=30000)
    smiles_list = df["canonical_smiles"].tolist()  # Adjust column name if different

    try:
        # Initialize the benchmark class
        # Use n_jobs=0 on Windows or for debugging DataLoader issues
        benchmark = NPGenBenchmark(
            device="cuda:0" if torch.cuda.is_available() else "cpu",
            n_jobs=4,  # Adjust based on your system cores
            batch_size=256,  # Example batch size
        )

        # Run the benchmark
        results = benchmark.benchmark(smiles_list)

        # Print results
        # Optionally convert to DataFrame
        results_df = pd.DataFrame(results)
        results_df.sort_values(by=["SMILES"], inplace=True)
        results_df.to_csv("benchmark_output.csv", index=False)

    except FileNotFoundError as e:
        print(f"\nError: A required file was not found: {e}")
        print("Please ensure model and index paths are correct.")
    except ImportError as e:
        print(f"\nError: Failed to import necessary modules: {e}")
        print(
            "Ensure 'npgenbenchmark' package structure is correct and dependencies are installed."
        )
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback

        traceback.print_exc()
