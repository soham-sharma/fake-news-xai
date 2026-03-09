import argparse
import pandas as pd

class ExperimentRunner:
    def __init__(self, config_path):
        pass
        
    def run_approach_a(self):
        print("Running DT models")
        
    def run_approach_b(self):
        print("Running BERT models")

if __name__ == "__main__":
    runner = ExperimentRunner("config.yaml")
    runner.run_approach_a()
    def setup_directories(self):
        # creating results and experiment nested folders
        pass

    def run_debiasing(self):
        from src.evaluation.debias import LexicalDebiaser
        print("auditing datasets and regenerating test combinations")

    def _snapshot_artifacts(self, artifact_paths):
        # storing model objects internally 
        pass
