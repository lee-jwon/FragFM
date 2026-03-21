from pathlib import Path

import pandas as pd
import torch
from npgenbenchmark import NPGenBenchmark

TEST_DIR = Path(__file__).parent

if __name__ == "__main__":
    input_file = TEST_DIR / "samples.csv"
    df = pd.read_csv(input_file, low_memory=False)
    smiles_list = df["canonical_smiles"].tolist()

    try:
        # Initialize the benchmark class
        # Use n_jobs=0 on Windows or for debugging DataLoader issues
        benchmark = NPGenBenchmark(
            device="cuda:0" if torch.cuda.is_available() else "cpu",
            n_jobs=4,  # Adjust based on your system cores
            batch_size=256,  # Example batch size
            verbose=True,
        )

        # Run per-SMILES NP classification (answer.csv-compatible format)
        results = benchmark.run_np_classifier(smiles_list)
        results_df = pd.DataFrame(results)
        results_df = results_df[
            [
                "SMILES",
                "Class_Results",
                "Superclass_Results",
                "Pathway_Results",
                "Is_Glycoside",
            ]
        ]
        results_df.sort_values(by=["SMILES"], inplace=True)
        results_df.to_csv("benchmark_output.csv", index=False)

    except FileNotFoundError as e:
        print(f"\nError: A required file was not found: {e}")
        print("Please ensure model and index paths are correct.")
        raise SystemExit(1)
    except ImportError as e:
        print(f"\nError: Failed to import necessary modules: {e}")
        print(
            "Ensure 'npgenbenchmark' package structure is correct and dependencies are installed."
        )
        raise SystemExit(1)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback

        traceback.print_exc()
        raise SystemExit(1)
