"""
Multi-Fidelity Neural Network Modeling Pipeline
Refactored for simplicity, readability, and reduced duplication
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

# Set random seed for reproducibility
random.seed(1)
np.random.seed(1)

class MultiFidelityPipeline:
    """Main pipeline for multi-fidelity modeling experiments"""
    
    def __init__(self):
        self.data = self._generate_data()
        
    @staticmethod
    def high_fidelity_func(x):
        """High fidelity function: ((6x-2)^2 * sin(12x-4))"""
        return ((6*x - 2)**2) * np.sin(12*x - 4)
    
    @staticmethod
    def low_fidelity_func(x, A=0.5, B=10, C=-5):
        """Low fidelity function with parameters A, B, C"""
        return A * MultiFidelityPipeline.high_fidelity_func(x) + B*(x-0.5) + C
    
    def _generate_data(self):
        """Generate training data for all fidelity levels"""
        X_low = np.linspace(0, 1.5, 20)
        X_med = np.linspace(0, 1.5, 10) 
        X_high = np.linspace(0, 1.5, 5)
        
        y_high = [self.high_fidelity_func(x) for x in X_high]
        y_med = [self.low_fidelity_func(x, A=0.8, B=2, C=-1) for x in X_med]
        y_low = [self.low_fidelity_func(x) for x in X_low]
        
        return {
            'X_low': X_low, 'X_med': X_med, 'X_high': X_high,
            'y_low': y_low, 'y_med': y_med, 'y_high': y_high
        }
    
    def create_model(self, hidden_layers=(20, 15, 10), activation='logistic'):
        """Create MLPRegressor with standard parameters"""
        return MLPRegressor(
            random_state=1,
            hidden_layer_sizes=hidden_layers,
            activation=activation,
            solver='adam',
            learning_rate='adaptive',
            tol=1e-4,
            max_iter=30000
        )
    
    def prepare_data(self, X, y):
        """Prepare data for training (reshape and flatten as needed)"""
        X = np.array(X).reshape(-1, 1)
        y = np.ravel(y)
        return X, y
    
    def generate_test_data(self, n_samples=20, x_range=(0, 1.5)):
        """Generate random test samples"""
        x_test = [random.uniform(*x_range) for _ in range(n_samples)]
        x_test = np.array(sorted(x_test)).reshape(-1, 1)
        y_test = np.array([self.high_fidelity_func(x) for x in x_test.flatten()])
        return x_test, y_test
    
    def evaluate_model(self, model, X_test, y_test):
        """Evaluate model performance with MSE and R²"""
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = model.score(X_test, y_test)
        return round(mse, 5), round(r2, 5)
    
    def plot_fidelities(self):
        """Plot all fidelity functions and training data"""
        x = np.linspace(0, 2, 1000)
        
        y_high = [self.high_fidelity_func(xi) for xi in x]
        y_med = [self.low_fidelity_func(xi, A=0.8, B=2, C=-1) for xi in x]
        y_low = [self.low_fidelity_func(xi) for xi in x]
        
        plt.figure(figsize=(10, 6))
        
        # Plot functions
        plt.plot(x, y_high, 'r-', alpha=0.8, label=r'$y_H$')
        plt.plot(x, y_med, 'orange', alpha=0.8, label=r'$y_M$')
        plt.plot(x, y_low, 'pink', alpha=0.8, label=r'$y_L$')
        
        # Plot training data
        plt.scatter(self.data['X_high'], self.data['y_high'], c='red', s=10, alpha=0.7, label=r'$D_H$')
        plt.scatter(self.data['X_med'], self.data['y_med'], c='orange', s=10, alpha=0.7, label=r'$D_M$')
        plt.scatter(self.data['X_low'], self.data['y_low'], c='pink', s=10, alpha=0.7, label=r'$D_L$')
        
        plt.xlabel('X')
        plt.ylabel('y')
        plt.grid(True, alpha=0.3)
        plt.xlim(-0.1, 1.6)
        plt.legend()
        plt.gca().set_facecolor((0.95, 0.95, 0.95))
        plt.title('Multi-Fidelity Functions and Training Data')
        plt.show()

class MultiFidelityModels:
    """Container for different multi-fidelity modeling approaches"""
    
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.data = pipeline.data
    
    def single_fidelity(self, fidelity='low'):
        """Train model on single fidelity data"""
        X, y = self.pipeline.prepare_data(
            self.data[f'X_{fidelity}'], 
            self.data[f'y_{fidelity}']
        )
        
        model = self.pipeline.create_model()
        model.fit(X, y)
        
        X_test, y_test = self.pipeline.generate_test_data()
        mse, r2 = self.pipeline.evaluate_model(model, X_test, y_test)
        
        print(f"{fidelity.capitalize()} fidelity - MSE: {mse}, R²: {r2}")
        return model
    
    def warm_learning(self, iterations=10):
        """Multi-Fidelity Warm Learning: progressively add higher fidelity data"""
        # Start with low fidelity
        X_low, y_low = self.pipeline.prepare_data(self.data['X_low'], self.data['y_low'])
        model = self.pipeline.create_model()
        model.fit(X_low, y_low)
        
        X_test, y_test = self.pipeline.generate_test_data()
        results = {'low': self.pipeline.evaluate_model(model, X_test, y_test)}
        
        # Add medium fidelity data
        X_med, y_med = self.pipeline.prepare_data(self.data['X_med'], self.data['y_med'])
        for _ in range(iterations):
            model.partial_fit(X_med, y_med)
        results['med'] = self.pipeline.evaluate_model(model, X_test, y_test)
        
        # Add high fidelity data
        X_high, y_high = self.pipeline.prepare_data(self.data['X_high'], self.data['y_high'])
        for _ in range(iterations):
            model.partial_fit(X_high, y_high)
        results['high'] = self.pipeline.evaluate_model(model, X_test, y_test)
        
        print("Warm Learning Results:")
        for level, (mse, r2) in results.items():
            print(f"  {level.capitalize()}: MSE={mse}, R²={r2}")
        
        return model, results
    
    def progressive_modeling(self):
        """Multi-Fidelity Progressive Modeling: stack predictions from lower fidelities"""
        X_test, y_test = self.pipeline.generate_test_data()
        
        # Train low fidelity model
        X_low, y_low = self.pipeline.prepare_data(self.data['X_low'], self.data['y_low'])
        model_low = self.pipeline.create_model()
        model_low.fit(X_low, y_low)
        
        # Train medium fidelity model (stacked with low predictions)
        X_med, y_med = self.pipeline.prepare_data(self.data['X_med'], self.data['y_med'])
        y_low_pred = model_low.predict(X_med)
        X_med_stacked = np.hstack([X_med, y_low_pred.reshape(-1, 1)])
        
        model_med = self.pipeline.create_model(hidden_layers=(10, 10))
        model_med.fit(X_med_stacked, y_med)
        
        # Train high fidelity model (stacked with low and medium predictions)
        X_high, y_high = self.pipeline.prepare_data(self.data['X_high'], self.data['y_high'])
        y_low_high = model_low.predict(X_high)
        X_high_with_low = np.hstack([X_high, y_low_high.reshape(-1, 1)])
        y_med_high = model_med.predict(X_high_with_low)
        X_high_stacked = np.hstack([X_high, y_med_high.reshape(-1, 1)])
        
        model_high = self.pipeline.create_model(hidden_layers=(10,), activation='relu')
        model_high.fit(X_high_stacked, y_high)
        
        # Evaluate on test data
        y_low_test = model_low.predict(X_test)
        X_test_with_low = np.hstack([X_test, y_low_test.reshape(-1, 1)])
        y_med_test = model_med.predict(X_test_with_low)
        X_test_stacked = np.hstack([X_test, y_med_test.reshape(-1, 1)])
        
        mse, r2 = self.pipeline.evaluate_model(model_high, X_test_stacked, y_test)
        print(f"Progressive Modeling - MSE: {mse}, R²: {r2}")
        
        return model_high, (model_low, model_med)
    
    def plot_comparison(self):
        """Plot comparison of different modeling approaches"""
        X_plot = np.linspace(0, 1.5, 1000).reshape(-1, 1)
        y_true = [self.pipeline.high_fidelity_func(x) for x in X_plot.flatten()]
        
        # Get predictions from different models
        model_low = self.single_fidelity('low')
        model_wl, _ = self.warm_learning()
        model_pm, (m_low, m_med) = self.progressive_modeling()
        
        y_low_pred = model_low.predict(X_plot)
        y_wl_pred = model_wl.predict(X_plot)
        
        # Progressive model predictions require stacking
        y_low_for_pm = m_low.predict(X_plot)
        X_with_low = np.hstack([X_plot, y_low_for_pm.reshape(-1, 1)])
        y_med_for_pm = m_med.predict(X_with_low)
        X_stacked = np.hstack([X_plot, y_med_for_pm.reshape(-1, 1)])
        y_pm_pred = model_pm.predict(X_stacked)
        
        plt.figure(figsize=(12, 8))
        
        # Plot training data
        plt.scatter(self.data['X_low'], self.data['y_low'], c='pink', s=10, alpha=0.5, label=r'$D_L$')
        plt.scatter(self.data['X_med'], self.data['y_med'], c='orange', s=10, alpha=0.5, label=r'$D_M$')
        plt.scatter(self.data['X_high'], self.data['y_high'], c='green', s=10, alpha=0.5, label=r'$D_H$')
        
        # Plot functions and predictions
        plt.plot(X_plot, y_true, 'r-', alpha=0.8, linewidth=2, label=r'$y_H$ (True)')
        plt.plot(X_plot, y_low_pred, 'pink', alpha=0.7, linewidth=2, label=r'$M_L$')
        plt.plot(X_plot, y_wl_pred, '--', color=(0.1, 0.1, 0.8), alpha=0.7, linewidth=2, label=r'$M_{H,WL}$')
        plt.plot(X_plot, y_pm_pred, '--', color=(0.1, 0.5, 0.8), alpha=0.7, linewidth=2, label=r'$M_{H,PM}$')
        
        plt.xlabel('X')
        plt.ylabel('y')
        plt.grid(True, alpha=0.3)
        plt.xlim(-0.1, 1.6)
        plt.legend()
        plt.gca().set_facecolor((0.95, 0.95, 0.95))
        plt.title('Multi-Fidelity Model Comparison')
        plt.show()

def main():
    """Main execution function"""
    # Initialize pipeline
    pipeline = MultiFidelityPipeline()
    
    # Show data visualization
    pipeline.plot_fidelities()
    
    # Initialize models
    models = MultiFidelityModels(pipeline)
    
    # Run different modeling approaches
    print("=== Single Fidelity Models ===")
    models.single_fidelity('low')
    models.single_fidelity('med')
    models.single_fidelity('high')
    
    print("\n=== Multi-Fidelity Approaches ===")
    models.warm_learning()
    models.progressive_modeling()
    
    # Show comparison plot
    models.plot_comparison()

if __name__ == "__main__":
    main()