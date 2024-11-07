import argparse
import os
import subprocess

def run_benchmark_commands(base_dir, output_dir, batch_sizes, prompt_lengths):
    """
    Generates and runs benchmark commands for each combination of batch size and prompt length.
    """
    for prompt_length in prompt_lengths:
        for batch_size in batch_sizes:
            # Define output file name based on batch size and prompt length
            output_file = os.path.join(output_dir, f"results_{prompt_length}_{batch_size}.json")
            
            # Construct the benchmark command
            command = [
                "python", "benchmark_e2e.py",
                "-i", base_dir,
                "-pm", "True",
                "-gc",
                "-b", str(batch_size),
                "-l", str(prompt_length),
                "-g", "256",
                "-r", "10",
                "-w", "2",
                "-k", "5",
                "-o", output_file
            ]
            
            # Run the command
            print(f"Running command: {' '.join(command)}")
            subprocess.run(command, check=True)

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Run benchmark tests with varying batch sizes and prompt lengths.")
    parser.add_argument("-b", "--batch_sizes", default=[1,4,16,32,64],  help="List of batch sizes to test.")
    parser.add_argument("-l", "--prompt_lengths", nargs="+",default=[16,64],  help="List of prompt lengths to test.")

    args = parser.parse_args()
# python benchmark_e2e.py -i "C:\Users\ofdeme\Documents\projects\onnx_spike\cuda\cuda-int4-rtn-block-32" -pm True -gc -b 64 -l 16 -g 256 -r 10 -w 2 -k 5 -o cuda_results\results_16_64.json
    input_dir="C:\\Users\\ofdeme\\Documents\\projects\\onnx_spike\\cuda\\cuda-fp16"
    output_path = f"cuda_results\fp16"

    # Run benchmarks
    run_benchmark_commands(
        base_dir=input_dir,
        output_dir=output_path,
        batch_sizes=args.batch_sizes,
        prompt_lengths=args.prompt_lengths
    )
