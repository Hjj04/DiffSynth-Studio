# tools/pick_best_checkpoint.py
#!/usr/bin/env python3
import os
import argparse
import pandas as pd

def find_metrics_csv(run_dir: str) -> str:
    """Finds the metrics.csv file in a given run directory."""
    for root, _, files in os.walk(run_dir):
        for file in files:
            if file == "metrics.csv":
                return os.path.join(root, file)
    raise FileNotFoundError(f"Could not find metrics.csv in or under {run_dir}")

def pick_best_checkpoint(metrics_csv: str, metric: str = "LPIPS", mode: str = "min") -> tuple[int, dict]:
    """Picks the best row from a metrics DataFrame based on a given metric and mode."""
    df = pd.read_csv(metrics_csv)
    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not in CSV columns: {df.columns.tolist()}")
    
    if mode == "min":
        best_row = df.loc[df[metric].idxmin()]
    elif mode == "max":
        best_row = df.loc[df[metric].idxmax()]
    else:
        raise ValueError(f"Mode must be 'min' or 'max', but got '{mode}'.")
        
    # Assuming there's a 'step' or similar column to identify the checkpoint
    step_col = next((col for col in df.columns if 'step' in col.lower()), None)
    if step_col is None:
        raise ValueError(f"Could not find a 'step' column in {metrics_csv}")
        
    return int(best_row[step_col]), best_row.to_dict()

def main():
    parser = argparse.ArgumentParser(description="Pick the best checkpoint from an experiment run based on metrics.")
    parser.add_argument("run_dir", help="The experiment run directory containing metrics.csv and outputs.")
    parser.add_argument("--metric", default="LPIPS", help="Metric to use for picking the best checkpoint (e.g., LPIPS, Edge_IoU, CLIP_Consistency).")
    parser.add_argument("--mode", default="min", choices=["min", "max"], help="Minimize or maximize the metric.")
    args = parser.parse_args()

    try:
        metrics_file = find_metrics_csv(args.run_dir)
        step, best_metrics = pick_best_checkpoint(metrics_file, args.metric, args.mode)
        
        # The training script saves outputs like `ground_truth.pt` in the main logdir
        checkpoint_path = os.path.join(args.run_dir, "predicted_frames.pt")
        if not os.path.exists(checkpoint_path):
             raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}. The script saves the final state, not per-step checkpoints.")
        
        print("--- Best Performing Step ---")
        print(f"Metric: {args.metric} ({args.mode})")
        for key, value in best_metrics.items():
            print(f"  - {key}: {value}")
        print("\n--- Corresponding Output File ---")
        print(checkpoint_path)

    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()