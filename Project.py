"""
Geotechnical Deflection Analysis Pipeline
Multi-fidelity modeling for predicting excavation-induced deflections
Refactored for clarity, maintainability, and reduced duplication
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt

class DeflectionAnalyzer:
    """Main class for geotechnical deflection analysis and modeling"""
    
    def __init__(self, exp_data_path='Site Data.csv', plaxis_data_path='All_def.csv', 
                 soil_params_path='SoilParameters.csv'):
        """Initialize with data paths and load all datasets"""
        self.depth_points = 29
        self.max_depth = 14.5
        self.depths = np.linspace(0, self.max_depth, self.depth_points)
        
        # Load and process data
        self.exp_data = self._load_experimental_data(exp_data_path)
        self.plaxis_data = self._load_plaxis_data(plaxis_data_path)
        self.soil_params = self._load_soil_parameters(soil_params_path)
        
        # Define true values and soil parameters
        self.site_params = [83800, 43.5, 7.7, 12400, 31.1]  # GO_E, GO_phi, TF_c, TF_E, TF_phi
        self.param_names = ['GO_E', 'GO_phi', 'TF_c', 'TF_E', 'TF_phi', 'stage']
        
        # Prepare datasets for modeling
        self.x_true = np.array(self.site_params + [11]).reshape(1, -1)
        self.y_true = np.array(self.exp_data.iloc[-1])  # Final stage deflection
        self.y_prev = np.array(self.exp_data.iloc[-2])  # Previous stage deflection
        
    def _load_experimental_data(self, path):
        """Load and process experimental deflection data"""
        exp = pd.read_csv(path, header=0, index_col=0)
        exp_abs = pd.DataFrame(abs(exp) / 1000).transpose()
        exp_abs.reset_index(drop=True, inplace=True)
        return {
            'x_plaxis': x_plaxis, 'y_plaxis': y_plaxis,
            'x_exp': x_exp, 'y_exp': y_exp
        }

class DeflectionModels:
    """Container for different modeling approaches"""
    
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.datasets = analyzer.prepare_datasets()
        self.models = {}
        self.predictions = {}
    
    def train_low_fidelity_model(self):
        """Train model on Plaxis simulation data only"""
        model = self.analyzer.create_model(hidden_layers=(25,))
        x_plx = self.datasets['x_plaxis']
        y_plx = self.datasets['y_plaxis']
        
        model.fit(x_plx, y_plx)
        prediction = model.predict(self.analyzer.x_true)
        
        self.models['low_fidelity'] = model
        self.predictions['low_fidelity'] = prediction.flatten()
        return model, prediction
    
    def train_combined_model(self, use_short_history=False):
        """Train model on combined Plaxis and experimental data"""
        x_plx, y_plx = self.datasets['x_plaxis'], self.datasets['y_plaxis']
        x_exp, y_exp = self.datasets['x_exp'], self.datasets['y_exp']
        
        if use_short_history:
            # Use only recent experimental data (excluding first few and last stages)
            x_exp_short = x_exp.iloc[6:-1]
            y_exp_short = y_exp.iloc[6:-1]
            x_combined = np.vstack([x_plx, x_exp_short])
            y_combined = np.vstack([y_plx, y_exp_short])
            model_key = 'combined_short'
        else:
            # Use all experimental data
            x_combined = np.vstack([x_plx, x_exp])
            y_combined = np.vstack([y_plx, y_exp])
            model_key = 'combined_full'
        
        model = self.analyzer.create_model(hidden_layers=(30, 30), max_iter=10000)
        model.fit(x_combined, y_combined)
        prediction = model.predict(self.analyzer.x_true)
        
        self.models[model_key] = model
        self.predictions[model_key] = prediction.flatten()
        return model, prediction
    
    def train_multifidelity_model(self):
        """Train multi-fidelity model using progressive approach"""
        # First train low-fidelity model
        low_model, _ = self.train_low_fidelity_model()
        
        # Get low-fidelity predictions for experimental inputs
        x_exp = self.datasets['x_exp']
        y_exp = self.datasets['y_exp']
        y_low_exp = low_model.predict(x_exp)
        
        # Stack experimental inputs with low-fidelity predictions
        x_stacked = np.hstack([x_exp, y_low_exp])
        
        # Train high-fidelity model
        high_model = self.analyzer.create_model(hidden_layers=(30,))
        high_model.fit(x_stacked, y_exp)
        
        # Make prediction for true case
        y_low_true = low_model.predict(self.analyzer.x_true)
        x_true_stacked = np.hstack([self.analyzer.x_true, y_low_true])
        prediction = high_model.predict(x_true_stacked)
        
        # Enhanced prediction using quadratic combination
        enhanced_prediction = 2.1 * prediction + 0.5 * prediction**2 + 0.1 * y_low_true
        
        self.models['multifidelity'] = (low_model, high_model)
        self.predictions['multifidelity'] = prediction.flatten()
        self.predictions['multifidelity_enhanced'] = enhanced_prediction.flatten()
        
        return (low_model, high_model), prediction
    
    def train_all_models(self):
        """Train all modeling approaches"""
        print("Training low-fidelity model...")
        self.train_low_fidelity_model()
        
        print("Training combined models...")
        self.train_combined_model(use_short_history=False)
        self.train_combined_model(use_short_history=True)
        
        print("Training multi-fidelity model...")
        self.train_multifidelity_model()
        
        print("All models trained successfully!")

class DeflectionVisualizer:
    """Visualization methods for deflection analysis"""
    
    def __init__(self, analyzer, models=None):
        self.analyzer = analyzer
        self.models = models
        self._setup_plot_style()
    
    def _setup_plot_style(self):
        """Set up consistent plot styling"""
        self.colors = {
            'true': (0.8, 0.2, 0.2),
            'previous': (0.9, 0.5, 0),
            'low_fidelity': (0.2, 0.6, 0.8),
            'combined_full': (0.2, 0.2, 0.8),
            'combined_short': (0.4, 0.2, 0.8),
            'multifidelity': (0.2, 0.7, 0.4),
            'experimental': (0.1, 0.2, 0.7)
        }
        
        self.plot_config = {
            'grid': True,
            'background': (0.95, 0.95, 0.95),
            'alpha': 0.6,
            'xlim': (0, 0.018)
        }
    
    def plot_experimental_curves(self, normalized=False):
        """Plot experimental deflection curves over time"""
        data = normalize(abs(self.analyzer.exp_data), axis=0).T if normalized else self.analyzer.exp_data
        title = 'Normalized' if normalized else 'Absolute'
        xlabel = 'Normalized Deflection' if normalized else 'Deflection (m)'
        
        plt.figure(figsize=(10, 8))
        
        for idx, row in data.iterrows():
            color = self.colors['true'] if idx == len(data)-1 else (0.1, 0.2+0.05*idx, 0.7)
            plt.plot(row, self.analyzer.depths, color=color, alpha=0.8)
        
        self._format_plot(xlabel, f'{title} deflection curves during excavation')
        plt.legend(range(1, len(data)+1), title=r'$\alpha$ (Stage)')
        plt.show()
    
    def plot_plaxis_vs_experimental(self, normalized=True):
        """Plot Plaxis simulation results against experimental data"""
        plt.figure(figsize=(10, 8))
        
        if normalized:
            plaxis_norm = normalize(self.analyzer.plaxis_data.values, axis=1)
            exp_norm = normalize(abs(self.analyzer.exp_data.values), axis=0).T
            
            # Plot Plaxis curves
            for i in range(min(11, len(plaxis_norm))):
                color = (0.2, 0.2+(0.5*i/len(plaxis_norm)), 0.8)
                plt.plot(plaxis_norm[i], self.analyzer.depths, color=color, alpha=0.6)
            
            # Plot final experimental curve
            plt.plot(exp_norm.iloc[-1], self.analyzer.depths, 
                    color=self.colors['true'], alpha=0.8, linewidth=2)
            
            xlabel, title = 'Normalized Deflection', 'Normalized deflection curves: Plaxis vs Experimental'
        else:
            # Plot absolute values
            for i in range(min(10, len(self.analyzer.plaxis_data))):
                color = (0.2, 0.2+(0.5*i/len(self.analyzer.plaxis_data)), 0.8)
                plt.plot(self.analyzer.plaxis_data.iloc[i], self.analyzer.depths, 
                        color=color, alpha=0.6)
            
            plt.plot(self.analyzer.y_true, self.analyzer.depths,
                    color=self.colors['true'], alpha=0.8, linewidth=2)
            
            xlabel, title = 'Deflection (m)', 'Absolute deflection curves: Plaxis vs Experimental'
        
        self._format_plot(xlabel, title)
        plt.show()
    
    def plot_model_predictions(self, models_to_plot=None):
        """Plot predictions from different models"""
        if not self.models:
            print("No models available. Train models first.")
            return
        
        if models_to_plot is None:
            models_to_plot = ['low_fidelity', 'combined_short', 'multifidelity_enhanced']
        
        plt.figure(figsize=(12, 8))
        
        # Plot predictions
        for model_name in models_to_plot:
            if model_name in self.models.predictions:
                prediction = self.models.predictions[model_name]
                color = self.colors.get(model_name, (0.5, 0.5, 0.5))
                label = self._get_model_label(model_name)
                plt.plot(prediction, self.analyzer.depths, color=color, 
                        alpha=0.8, linewidth=2, label=label)
        
        # Plot true data
        plt.plot(self.analyzer.y_true, self.analyzer.depths, 
                color=self.colors['true'], alpha=0.8, linewidth=2, 
                linestyle='--', label=r'$y^*$ (True)')
        
        self._format_plot('Deflection (m)', 'Model Predictions vs True Deflection')
        plt.legend()
        plt.show()
    
    def plot_experimental_stages(self, show_plaxis=True):
        """Plot experimental deflection progression through excavation stages"""
        plt.figure(figsize=(12, 8))
        
        if show_plaxis and 'low_fidelity' in self.models.predictions:
            plt.plot(self.models.predictions['low_fidelity'], self.analyzer.depths,
                    color=(0.9, 0.3, 0.3), alpha=0.6, linewidth=2, label='Plaxis2D Prediction')
        
        # Plot first 8 experimental stages with gradient colors
        for i in range(min(8, len(self.analyzer.exp_data))):
            color = (0.2 + 0.05*i, 0.4 + 0.05*i, 0.9)
            plt.plot(self.analyzer.exp_data.iloc[i], self.analyzer.depths,
                    color=color, alpha=0.6, label=f'Stage {i+1}')
        
        self._format_plot('Deflection (m)', 'Deflection Evolution Through Excavation Stages')
        plt.legend(loc='upper right')
        plt.show()
    
    def plot_final_comparison(self):
        """Plot final true deflection curve"""
        plt.figure(figsize=(10, 8))
        
        plt.plot(self.analyzer.y_true, self.analyzer.depths, 
                linestyle='--', color='purple', alpha=0.8, linewidth=3,
                label='True Final Deflection')
        
        self._format_plot('Deflection (m)', 'Final Excavation Deflection Profile')
        plt.legend(loc='upper right')
        plt.show()
    
    def _get_model_label(self, model_name):
        """Get formatted label for model"""
        labels = {
            'low_fidelity': r'$\hat{y}^*_{L}$ (Low Fidelity)',
            'combined_full': r'$\hat{y}^*_{G}$ (Combined Full)',
            'combined_short': r'$\hat{y}^*_{S}$ (Combined Short)',
            'multifidelity': r'$\hat{y}^*_{MF}$ (Multi-Fidelity)',
            'multifidelity_enhanced': r'$\hat{y}^*$ (Enhanced MF)'
        }
        return labels.get(model_name, model_name)
    
    def _format_plot(self, xlabel, title):
        """Apply consistent formatting to plots"""
        plt.gca().invert_yaxis()
        plt.xlabel(xlabel)
        plt.ylabel('Depth (m)')
        plt.title(title)
        plt.grid(self.plot_config['grid'], alpha=0.3)
        plt.xlim(self.plot_config['xlim'])
        plt.gca().set_facecolor(self.plot_config['background'])

def main():
    """Main execution pipeline"""
    # Initialize analyzer
    print("Loading and processing data...")
    analyzer = DeflectionAnalyzer()
    
    # Initialize models
    models = DeflectionModels(analyzer)
    
    # Train all models
    models.train_all_models()
    
    # Initialize visualizer
    viz = DeflectionVisualizer(analyzer, models)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    
    # Show data overview
    viz.plot_experimental_curves(normalized=False)
    viz.plot_plaxis_vs_experimental(normalized=True)
    
    # Show model comparisons
    viz.plot_model_predictions()
    viz.plot_experimental_stages(show_plaxis=True)
    viz.plot_final_comparison()
    
    print("\nAnalysis complete!")
    
    # Print model performance summary
    print("\nModel Predictions Summary:")
    for name, prediction in models.predictions.items():
        max_def = np.max(prediction)
        print(f"{name}: Max deflection = {max_def:.6f} m")

if __name__ == "__main__":
    main()
    
    def _load_plaxis_data(self, path):
        """Load and process Plaxis simulation data"""
        plx = np.genfromtxt(path, delimiter=',')
        plx = [row[1:] for row in plx]  # Remove depth indicators
        
        # Sample to match experimental data points
        indexes = np.linspace(0, len(plx[0])-1, self.depth_points, dtype=int)
        plx_sampled = [[row[index] for index in indexes] for row in plx]
        
        return pd.DataFrame(plx_sampled)
    
    def _load_soil_parameters(self, path):
        """Load soil parameters used in Plaxis simulations"""
        return pd.read_csv(path, names=self.param_names[:-1])
    
    def create_model(self, hidden_layers=(30,), activation='logistic', solver='lbfgs', 
                    max_iter=1000, tol=1e-4):
        """Create MLPRegressor with specified parameters"""
        return MLPRegressor(
            random_state=1,
            hidden_layer_sizes=hidden_layers,
            activation=activation,
            solver=solver,
            max_iter=max_iter,
            tol=tol,
            learning_rate='adaptive'
        )
    
    def prepare_datasets(self):
        """Prepare input/output datasets for different modeling approaches"""
        # Plaxis dataset
        plx_stages = pd.DataFrame(11 * np.ones(len(self.soil_params)), dtype=int)
        x_plaxis = pd.concat([self.soil_params, plx_stages, self.plaxis_data], 
                           axis=1, ignore_index=True).iloc[:, :6]
        y_plaxis = self.plaxis_data
        
        # Experimental dataset (all stages)
        exp_inputs = []
        for idx, deflections in self.exp_data.iterrows():
            stage_input = self.site_params + [idx + 1]
            exp_inputs.append(stage_input)
        
        x_exp = pd.DataFrame(exp_inputs)
        y_exp = self.exp_data
        
        return