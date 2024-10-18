import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Union

class ProjectRiskFuzzyBWM:
    """
    Fuzzy BWM implementation for Project Management Risk Assessment
    """
    def __init__(self):
        self.fuzzy_scale = {
            'EI': (1, 1, 1),        # Equal Importance
            'WMI': (2/3, 1, 3/2),   # Weakly More Important
            'SMI': (3/2, 2, 5/2),   # Strongly More Important
            'VSMI': (5/2, 3, 7/2),  # Very Strongly More Important
            'AMI': (7/2, 4, 9/2)    # Absolutely More Important
        }
        
        self.criteria = None
        self.n_criteria = None
        self.best_criterion = None
        self.worst_criterion = None
        self.results = {}
        
    def set_problem(self, criteria: List[str], best: str, worst: str) -> None:
        """Set up the risk assessment problem"""
        if not criteria or len(criteria) < 2:
            raise ValueError("At least two risk criteria are required")
        if len(set(criteria)) != len(criteria):
            raise ValueError("Risk criteria names must be unique")
        if best not in criteria or worst not in criteria:
            raise ValueError("Best and worst criteria must be in the criteria list")
            
        self.criteria = criteria
        self.n_criteria = len(criteria)
        self.best_criterion = best
        self.worst_criterion = worst
        
    def defuzzify(self, fuzzy_number: Tuple[float, float, float]) -> float:
        """Convert triangular fuzzy number to crisp number"""
        return sum(fuzzy_number) / 3
        
    def calculate_weights(self, best_to_others: List[str], others_to_worst: List[str]) -> Dict:
        """Calculate risk criteria weights"""
        # Convert preferences to fuzzy numbers
        bo_fuzzy = [self.fuzzy_scale[x] for x in best_to_others]
        ow_fuzzy = [self.fuzzy_scale[x] for x in others_to_worst]
        
        # Defuzzify
        bo_crisp = [self.defuzzify(x) for x in bo_fuzzy]
        ow_crisp = [self.defuzzify(x) for x in ow_fuzzy]
        
        # Initialize weights
        weights = np.ones(self.n_criteria) / self.n_criteria
        best_idx = self.criteria.index(self.best_criterion)
        worst_idx = self.criteria.index(self.worst_criterion)
        
        # Iterative weight calculation
        for _ in range(1000):
            prev_weights = weights.copy()
            
            # Update weights
            for i in range(self.n_criteria):
                if i != best_idx:
                    weights[i] = weights[best_idx] / bo_crisp[i]
                if i != worst_idx:
                    weights[i] = ow_crisp[i] * weights[worst_idx]
            
            # Normalize
            weights = weights / np.sum(weights)
            
            # Check convergence
            if np.all(np.abs(weights - prev_weights) < 1e-6):
                break
                
        self.results = {'weights': weights}
        return self.results
    
    def visualize_results(self, save_path: str = None):
        """Visualize risk assessment results"""
        if not self.results:
            raise ValueError("Weights haven't been calculated yet")
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Create bar plot
        bars = plt.bar(self.criteria, self.results['weights'])
        
        # Customize plot
        plt.title('Project Management Risk Criteria Weights', pad=20)
        plt.xlabel('Risk Criteria')
        plt.ylabel('Weight')
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()

def run_project_risk_assessment():
    """
    Run a complete project risk assessment example
    """
    # Initialize
    risk_assessment = ProjectRiskFuzzyBWM()
    
    # Define risk criteria
    risk_criteria = [
        'Technical Risk',          # TR: Risks related to technology and technical requirements
        'Schedule Risk',           # SR: Risks affecting project timeline
        'Cost Risk',               # CR: Risks affecting project budget
        'Resource Risk',           # RR: Risks related to human and material resources
        'Scope Risk',              # ScR: Risks related to project scope changes
        'Quality Risk',            # QR: Risks affecting deliverable quality
        'Stakeholder Risk',        # StR: Risks related to stakeholder management
        'Communication Risk'       # ComR: Risks in project communication
    ]
    
    # Set best and worst criteria based on expert judgment
    best_criterion = 'Technical Risk'  # Most important risk to consider
    worst_criterion = 'Communication Risk'  # Least important risk to consider
    
    # Set up the problem
    risk_assessment.set_problem(risk_criteria, best_criterion, worst_criterion)
    
    # Define best-to-others preferences
    # Technical Risk compared to others
    best_to_others = [
        'EI',    # Technical Risk to Technical Risk
        'WMI',   # Technical Risk to Schedule Risk
        'SMI',   # Technical Risk to Cost Risk
        'VSMI',  # Technical Risk to Resource Risk
        'SMI',   # Technical Risk to Scope Risk
        'WMI',   # Technical Risk to Quality Risk
        'VSMI',  # Technical Risk to Stakeholder Risk
        'AMI'    # Technical Risk to Communication Risk
    ]
    
    # Others-to-worst preferences
    # Other risks compared to Communication Risk
    others_to_worst = [
        'AMI',   # Technical Risk to Communication Risk
        'VSMI',  # Schedule Risk to Communication Risk
        'VSMI',  # Cost Risk to Communication Risk
        'SMI',   # Resource Risk to Communication Risk
        'SMI',   # Scope Risk to Communication Risk
        'WMI',   # Quality Risk to Communication Risk
        'WMI',   # Stakeholder Risk to Communication Risk
        'EI'     # Communication Risk to Communication Risk
    ]
    
    # Calculate weights
    results = risk_assessment.calculate_weights(best_to_others, others_to_worst)
    
    # Print detailed results
    print("\nProject Management Risk Assessment Results")
    print("=" * 50)
    print("\nRisk Criteria Weights:")
    print("-" * 30)
    
    # Create a formatted results table
    results_df = pd.DataFrame({
        'Risk Criterion': risk_criteria,
        'Weight': results['weights'],
        'Percentage': results['weights'] * 100
    })
    
    # Sort by weight in descending order
    results_df = results_df.sort_values('Weight', ascending=False)
    
    # Print formatted results
    print(results_df.to_string(float_format=lambda x: '{:.4f}'.format(x)))
    
    # Print risk priority levels
    print("\nRisk Priority Levels:")
    print("-" * 30)
    
    # Define priority levels
    def get_priority_level(weight):
        if weight >= 0.20: return "Very High"
        elif weight >= 0.15: return "High"
        elif weight >= 0.10: return "Medium"
        elif weight >= 0.05: return "Low"
        else: return "Very Low"
    
    results_df['Priority Level'] = results_df['Weight'].apply(get_priority_level)
    print(results_df[['Risk Criterion', 'Priority Level']])
    
    # Visualize results
    risk_assessment.visualize_results()
    
    # Additional visualization: Risk Matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        np.array(results['weights']).reshape((2, 4)),
        annot=True,
        fmt='.4f',
        cmap='YlOrRd',
        xticklabels=risk_criteria[4:],
        yticklabels=risk_criteria[:4]
    )
    plt.title('Risk Weight Matrix')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_project_risk_assessment()