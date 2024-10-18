import numpy as np

class FuzzyBWM:
    def __init__(self):
        self.criteria = None
        self.n_criteria = None
        self.best_criterion = None
        self.worst_criterion = None
        
    def set_criteria(self, criteria):
        """Set the criteria for the decision problem"""
        self.criteria = criteria
        self.n_criteria = len(criteria)
        
    def set_best_worst(self, best, worst):
        """Set the best and worst criteria"""
        self.best_criterion = best
        self.worst_criterion = worst
        
    def triangular_fuzzy_number(self, linguistic_term):
        """Convert linguistic terms to triangular fuzzy numbers"""
        fuzzy_scale = {
            'EI': (1, 1, 1),      # Equal Importance
            'WMI': (2/3, 1, 3/2), # Weakly More Important
            'SMI': (3/2, 2, 5/2), # Strongly More Important
            'VSMI': (5/2, 3, 7/2),# Very Strongly More Important
            'AMI': (7/2, 4, 9/2)  # Absolutely More Important
        }
        return fuzzy_scale.get(linguistic_term)
    
    def defuzzify(self, fuzzy_number):
        """Convert triangular fuzzy number to crisp number using centroid method"""
        l, m, u = fuzzy_number
        return (l + m + u) / 3
    
    def calculate_weights(self, best_to_others, others_to_worst):
        """Calculate the final weights using Fuzzy BWM"""
        # Initialize the model matrices
        n = self.n_criteria
        
        # Convert linguistic terms to fuzzy numbers
        best_to_others_fuzzy = [self.triangular_fuzzy_number(x) for x in best_to_others]
        others_to_worst_fuzzy = [self.triangular_fuzzy_number(x) for x in others_to_worst]
        
        # Defuzzify the preference values
        best_to_others_crisp = [self.defuzzify(x) for x in best_to_others_fuzzy]
        others_to_worst_crisp = [self.defuzzify(x) for x in others_to_worst_fuzzy]
        
        # Create the optimization problem
        weights = np.ones(n) / n  # Initial weights
        
        # Simple iterative method to approximate weights
        for _ in range(100):  # Number of iterations
            # Update weights based on best-to-others comparisons
            for i in range(n):
                if i != self.criteria.index(self.best_criterion):
                    weights[i] = weights[self.criteria.index(self.best_criterion)] / best_to_others_crisp[i]
            
            # Update weights based on others-to-worst comparisons
            worst_idx = self.criteria.index(self.worst_criterion)
            for i in range(n):
                if i != worst_idx:
                    weights[i] = others_to_worst_crisp[i] * weights[worst_idx]
            
            # Normalize weights
            weights = weights / np.sum(weights)
        
        return weights

# Example usage
def run_example():
    # Initialize the Fuzzy BWM solver
    fbwm = FuzzyBWM()
    
    # Define criteria
    criteria = ['C1', 'C2', 'C3', 'C4']
    fbwm.set_criteria(criteria)
    
    # Set best and worst criteria
    fbwm.set_best_worst('C1', 'C4')
    
    # Define best-to-others preferences (linguistic terms)
    best_to_others = ['EI', 'WMI', 'SMI', 'VSMI']
    
    # Define others-to-worst preferences (linguistic terms)
    others_to_worst = ['VSMI', 'SMI', 'WMI', 'EI']
    
    # Calculate weights
    weights = fbwm.calculate_weights(best_to_others, others_to_worst)
    
    # Print results
    print("\nFuzzy BWM Results:")
    print("=================")
    print("Criteria:", criteria)
    print("Best Criterion:", fbwm.best_criterion)
    print("Worst Criterion:", fbwm.worst_criterion)
    print("\nCalculated Weights:")
    for c, w in zip(criteria, weights):
        print(f"{c}: {w:.4f}")

if __name__ == "__main__":
    run_example()