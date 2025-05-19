import subprocess
import signal
import sys
import numpy as np
from typing import List
import time
import os
def run_evaluation(
    gpu_id: int,
    model_name: str,
    ratio: float,
    selection: int,  # New parameter
    mode: str = "fasttest",
    task_set: str = "transfer",
    prompt_method: str = "prompteol",
    contra: bool = True
) -> subprocess.CompletedProcess:
    """Run a single evaluation with the given parameters."""
    cmd = [
        "python", "evaluation.py",
        "--model_name_or_path", model_name,
        "--mode", mode,
        "--task_set", task_set,
        "--prompt_method", prompt_method,
        "--ratio", str(ratio),
        "--selection", str(selection)  # Add selection to command
    ]
    
    if contra:
        cmd.append("--contra")
    
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    return subprocess.run(
        cmd,
        env=env,
        check=False,
        text=True
    )

class EvaluationManager:
    def __init__(self):
        self.current_process = None
        self.setup_signal_handlers()
    
    def setup_signal_handlers(self):
        """Set up handlers for graceful termination."""
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    def signal_handler(self, signum, frame):
        """Handle termination signals by cleaning up current process."""
        if self.current_process:
            print("\nTerminating current process...")
            self.current_process.terminate()
            try:
                self.current_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print("Process didn't terminate gracefully, forcing...")
                self.current_process.kill()
        sys.exit(1)
    def run_evaluations_for_model_and_prompt(
        self,
        gpu_id: int,
        model: str,
        prompt_method: str,
        ratios: List[float],
        selections: List[int]  # New parameter
    ):
        """Run evaluations for a single model and prompt method combination."""
      
        # try:
        #     self.current_process = run_evaluation(
        #         gpu_id=gpu_id,
        #         model_name=model,
        #         ratio=0.0,
        #         selection=0,  # Default selection for non-contra
        #         prompt_method=prompt_method,
        #         contra=False
        #     )
        #     if self.current_process.returncode != 0:
        #         print(f"Warning: Non-contra process returned non-zero exit code: {self.current_process.returncode}")
        # except Exception as e:
        #     print(f"Error running non-contra evaluation: {e}")
        #     return
            
        # Then run with contra for each ratio and selection combination
        for selection in selections:
            for ratio in ratios:
                print(f"\nRunning evaluation with --contra flag (ratio = {ratio}, selection = {selection})...")
                try:
                    self.current_process = run_evaluation(
                        gpu_id=gpu_id,
                        model_name=model,
                        ratio=ratio,
                        selection=selection,
                        prompt_method=prompt_method,
                        contra=True
                    )
                    if self.current_process.returncode != 0:
                        print(f"Warning: Contra process returned non-zero exit code: {self.current_process.returncode}")
                except Exception as e:
                    print(f"Error running contra evaluation: {e}")
                
    def run_all_evaluations(
        self,
        gpu_id: int,
        models: List[str],
        prompt_methods: List[str],
        ratios: List[float],
        selections: List[int]  # New parameter
    ):
        """Run evaluations for all combinations of models and prompt methods."""
        total_combinations = len(models) * len(prompt_methods)
        current = 0
        
        for model in models:
            # if model == "tiiuae/falcon-7B":
            #     ratios = [0.1]
            #     selections = [2]
            # if model == "meta-llama/Llama-2-7b-hf":
            #     ratios = [1.0]
            #     selections = [3]
            for prompt_method in prompt_methods:
                current += 1
                print(f"\nProcessing combination {current}/{total_combinations}")
                self.run_evaluations_for_model_and_prompt(
                    gpu_id=gpu_id,
                    model=model,
                    prompt_method=prompt_method,
                    ratios=ratios,
                    selections=selections
                )

def main():
    # Configuration
    gpu_id = 0
    # prompt_methods = [
    prompt_methods = ["prompteol"]

    # models = [
    #     

    #     
    # ]
    models = [
        # "tiiuae/falcon-7B",
        "meta-llama/Meta-Llama-3-8B" , 
        # "huggyllama/llama-7b",
        # "meta-llama/Llama-2-7b-hf", 
        # "mistralai/Mistral-7B-v0.1"
        # "EleutherAI/pythia-6.9b"
        # "facebook/opt-6.7b", 
        # "facebook/opt-2.7b"
    ]
    ratios = [
        # -1.0,
        # -0.5,
        # # 0.5, 
        # # 1.0,
        # 1.0,
        # 3.0, 
        # 4.0,
        # -0.1,
        # -0.2,
        # -0.3,
        # 0.0,
        # 0.1, 
        0.5,
        # 0.3,
        # 0.4,
        # 0.5,
        # 0.6,
        # 0.7,
        # 0.8,
        # 0.9,
        # 1.0
        # 8.0
        
    ]
    
    selections = [3]  # New configuration
    
    # Create and run manager
    manager = EvaluationManager()
    manager.run_all_evaluations(
        gpu_id=gpu_id, 
        models=models, 
        prompt_methods=prompt_methods, 
        ratios=ratios,
        selections=selections
    )
    
if __name__ == "__main__":
    main()
