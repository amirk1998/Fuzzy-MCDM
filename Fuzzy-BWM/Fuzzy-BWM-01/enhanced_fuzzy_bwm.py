import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Union
import seaborn as sns

class EnhancedFuzzyBWM:
    """
    Enhanced implementation of Fuzzy Best-Worst Method (BWM) for Multi-Criteria Decision Making
    """
    
    def __init__(self):
        # Initialize fuzzy scale with triangular fuzzy numbers
        self.fuzzy_scale = {
            'EI': (1, 1, 1),        # Equal Importance
            'WMI': (2/3, 1, 3/2),   # Weakly More Important
            'SMI': (3/2, 2, 5/2),   # Strongly More Important
            'VSMI': (5/2, 3, 7/2),  # Very Strongly More Important
            'AMI': (7/2, 4, 9/2),   # Absolutely More Important
        }
        
        self.criteria = None
        self.n_criteria = None
        self.best_criterion = None
        self.worst_criterion = None
        self.results = {}
        
    def validate_inputs(self, criteria: List[str], best: str, worst: str) -> None:
        """Validate input parameters"""
        if not criteria or len(criteria) < 2:
            raise ValueError("At least two criteria are required")
        if len(set(criteria)) != len(criteria):
            raise ValueError("Criteria names must be unique")
        if best not in criteria or worst not in criteria:
            raise ValueError("Best and worst criteria must be in the criteria list")
        if best == worst:
            raise ValueError("Best and worst criteria cannot be the same")

    def set_problem(self, criteria: List[str], best: str, worst: str) -> None:
        """Set up the decision problem"""
        self.validate_inputs(criteria, best, worst)
        self.criteria = criteria
        self.n_criteria = len(criteria)
        self.best_criterion = best
        self.worst_criterion = worst
        
    def defuzzify(self, fuzzy_number: Tuple[float, float, float]) -> float:
        """Convert triangular fuzzy number to crisp number using centroid method"""
        return sum(fuzzy_number) / 3

    def consistency_ratio(self, weights: np.ndarray, best_to_others: List[str], 
                         others_to_worst: List[str]) -> float:
        """Calculate consistency ratio of the solution"""
        best_idx = self.criteria.index(self.best_criterion)
        worst_idx = self.criteria.index(self.worst_criterion)
        
        # Convert linguistic terms to crisp numbers
        bo_crisp = [self.defuzzify(self.fuzzy_scale[x]) for x in best_to_others]
        ow_crisp = [self.defuzzify(self.fuzzy_scale[x]) for x in others_to_worst]
        
        # Calculate consistency index
        max_deviation = 0
        for i in range(self.n_criteria):
            if i != best_idx:
                deviation = abs(weights[best_idx]/weights[i] - bo_crisp[i])
                max_deviation = max(max_deviation, deviation)
            if i != worst_idx:
                deviation = abs(weights[i]/weights[worst_idx] - ow_crisp[i])
                max_deviation = max(max_deviation, deviation)
                
        return max_deviation

    def calculate_weights(self, best_to_others: List[str], 
                         others_to_worst: List[str], max_iterations: int = 1000) -> Dict:
        """
        Calculate criteria weights using Fuzzy BWM
        """
        # Validate input preferences
        if len(best_to_others) != self.n_criteria or len(others_to_worst) != self.n_criteria:
            raise ValueError("Preference vectors must match the number of criteria")
            
        # Convert linguistic terms to fuzzy numbers
        bo_fuzzy = [self.fuzzy_scale[x] for x in best_to_others]
        ow_fuzzy = [self.fuzzy_scale[x] for x in others_to_worst]
        
        # Defuzzify the preference values
        bo_crisp = [self.defuzzify(x) for x in bo_fuzzy]
        ow_crisp = [self.defuzzify(x) for x in ow_fuzzy]
        
        # Initialize weights
        weights = np.ones(self.n_criteria) / self.n_criteria
        best_idx = self.criteria.index(self.best_criterion)
        worst_idx = self.criteria.index(self.worst_criterion)
        
        # Iterative weight calculation
        prev_weights = np.zeros(self.n_criteria)
        iteration = 0
        convergence_threshold = 1e-6
        
        while iteration < max_iterations:
            # Store previous weights
            prev_weights = weights.copy()
            
            # Update weights based on best-to-others comparisons
            for i in range(self.n_criteria):
                if i != best_idx:
                    weights[i] = weights[best_idx] / bo_crisp[i]
                    
            # Update weights based on others-to-worst comparisons
            for i in range(self.n_criteria):
                if i != worst_idx:
                    weights[i] = ow_crisp[i] * weights[worst_idx]
                    
            # Normalize weights
            weights = weights / np.sum(weights)
            
            # Check convergence
            if np.all(np.abs(weights - prev_weights) < convergence_threshold):
                break
                
            iteration += 1
        
        # Calculate consistency ratio
        cr = self.consistency_ratio(weights, best_to_others, others_to_worst)
        
        # Store results
        self.results = {
            'weights': weights,
            'consistency_ratio': cr,
            'iterations': iteration
        }
        
        return self.results

    def visualize_weights(self, save_path: str = None) -> None:
        """Visualize the calculated weights"""
        if not self.results:
            raise ValueError("Weights haven't been calculated yet")
            
        plt.figure(figsize=(10, 6))
        bars = plt.bar(self.criteria, self.results['weights'])
        plt.title('Criteria Weights from Fuzzy BWM')
        plt.xlabel('Criteria')
        plt.ylabel('Weight')
        plt.xticks(rotation=45)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}', ha='center', va='bottom')
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()

def run_example():
    """
    Run a comprehensive example of Fuzzy BWM
    """
    # Initialize the solver
    fbwm = EnhancedFuzzyBWM()
    
    # Define problem - Supplier Selection Example
    criteria = [
        'Quality',           # C1
        'Price',            # C2
        'Delivery Time',    # C3
        'Technology Level', # C4
        'Service Quality'   # C5
    ]
    
    best_criterion = 'Quality'
    worst_criterion = 'Service Quality'
    
    # Set up the problem
    fbwm.set_problem(criteria, best_criterion, worst_criterion)
    
    # Best-to-Others preferences
    best_to_others = [
        'EI',    # Quality to Quality
        'WMI',   # Quality to Price
        'SMI',   # Quality to Delivery Time
        'VSMI',  # Quality to Technology Level
        'AMI'    # Quality to Service Quality
    ]
    
    # Others-to-Worst preferences
    others_to_worst = [
        'AMI',   # Quality to Service Quality
        'VSMI',  # Price to Service Quality
        'SMI',   # Delivery Time to Service Quality
        'WMI',   # Technology Level to Service Quality
        'EI'     # Service Quality to Service Quality
    ]
    
    # Calculate weights
    results = fbwm.calculate_weights(best_to_others, others_to_worst)
    
    # Print detailed results
    print("\nFuzzy BWM Results:")
    print("=" * 50)
    print(f"Number of criteria: {fbwm.n_criteria}")
    print(f"Best criterion: {fbwm.best_criterion}")
    print(f"Worst criterion: {fbwm.worst_criterion}")
    print("\nPreference Information:")
    print("-" * 30)
    print("\nBest-to-Others preferences:")
    for c, p in zip(criteria, best_to_others):
        print(f"{fbwm.best_criterion} to {c}: {p}")
    print("\nOthers-to-Worst preferences:")
    for c, p in zip(criteria, others_to_worst):
        print(f"{c} to {fbwm.worst_criterion}: {p}")
    
    print("\nResults:")
    print("-" * 30)
    print("\nCriteria Weights:")
    for c, w in zip(criteria, results['weights']):
        print(f"{c}: {w:.4f}")
    print(f"\nConsistency Ratio: {results['consistency_ratio']:.4f}")
    print(f"Number of iterations: {results['iterations']}")
    
    # Visualize the results
    fbwm.visualize_weights()

if __name__ == "__main__":
    run_example()