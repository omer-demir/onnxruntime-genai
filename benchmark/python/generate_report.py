import os
import json
import pandas as pd

def extract_data_from_json(file_path, device, precision):
    """Reads a JSON file and extracts required fields for the report."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    rows = []
    for entry in data:
        config = entry["config"]
        metrics = entry["metrics"]
        
        row = {
            "device": device,
            "batch_size": config["batch_size"],
            "precision": precision,
            "prompt_length": config["prompt_length"],
            "tokens_generated": config["tokens_generated"],
            "max_length": config["max_length"],
            "token_generation_throughput_tps": str(metrics["token_generation_throughput_tps"]).replace('.',","),
            "token_generation_latency_ms": str(metrics["token_generation_latency_ms"]).replace('.',","),
            "wall_clock_throughput_tps": str(metrics["wall_clock_throughput_tps"]).replace('.',","),
            "wall_clock_time_s": str(metrics["wall_clock_time_s"]).replace('.',","),
            "peak_cpu_memory_gb": metrics.get("peak_cpu_memory_gb", None),
            "peak_gpu_memory_gb": metrics.get("peak_gpu_memory_gb", None)
        }
        rows.append(row)
    return rows

def get_precision(file_path, base_dir):
    """Determines the precision based on the directory structure."""
    if "cpu_results" in file_path:
        return "int4"
    elif "cuda_results" in file_path:
        if "fp16" in file_path:
            return "fp16"
        else:
            return "int4"
    return "int4"  # default precision

def process_directory(base_dir):
    """Processes all JSON files in a given directory to create a benchmark report."""
    all_rows = []
    for root, _, files in os.walk(base_dir):
        for file_name in files:
            if file_name.endswith('.json'):
                file_path = os.path.join(root, file_name)
                device = "cpu" if "cpu_results" in root else "cuda"
                precision = get_precision(file_path, base_dir)
                
                rows = extract_data_from_json(file_path, device, precision)
                all_rows.extend(rows)
    
    return all_rows

# Set paths to cpu_results and cuda_results
cpu_results_dir = 'cpu_results'
cuda_results_dir = 'cuda_results'

# Process each directory and gather data
cpu_data = process_directory(cpu_results_dir)
cuda_data = process_directory(cuda_results_dir)

# Combine all data
all_data = cpu_data + cuda_data

# Convert to DataFrame and save as CSV
df = pd.DataFrame(all_data)
df.to_csv('benchmark_report.csv', index=False)
print("Benchmark report saved as 'benchmark_report.csv'")
