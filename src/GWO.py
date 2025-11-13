import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
import copy
import time
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any
import logging
from .lstm_star import LSTM_STAR
from .utils import *
import traceback
class GreyWolfOptimizer:
    """
    Grey Wolf Optimizer for hyperparameter tuning of neural networks.
    Based on Mirjalili et al. (2014) with enhancements for neural network optimization.
    """
    
    def __init__(self, 
                 dim: int,
                 bounds: List[Tuple[float, float]],
                 n_wolves: int = 30,
                 max_iter: int = 100,
                 random_seed: int = None):
        """
        Initialize Grey Wolf Optimizer
        
        Args:
            dim: Dimension of search space (number of hyperparameters)
            bounds: List of (min, max) bounds for each hyperparameter
            n_wolves: Number of wolves in the pack (population size)
            max_iter: Maximum number of iterations
            random_seed: Random seed for reproducibility
        """
        self.dim = dim
        self.bounds = np.array(bounds)
        self.n_wolves = n_wolves
        self.max_iter = max_iter
        
        if random_seed:
            np.random.seed(random_seed)
            
        # Initialize wolf positions
        self.positions = self._initialize_population()
        self.fitness_values = np.full(n_wolves, float('inf'))
        
        # Best solutions (Alpha, Beta, Delta)
        self.alpha_pos = np.zeros(dim)
        self.alpha_score = float('inf')
        self.beta_pos = np.zeros(dim)
        self.beta_score = float('inf')
        self.delta_pos = np.zeros(dim)
        self.delta_score = float('inf')
        
        # Convergence tracking
        self.convergence_curve = []
        
    def _initialize_population(self) -> np.ndarray:
        """Initialize wolf population within bounds"""
        population = np.zeros((self.n_wolves, self.dim))
        for i in range(self.n_wolves):
            for j in range(self.dim):
                population[i, j] = np.random.uniform(
                    self.bounds[j, 0], self.bounds[j, 1]
                )
        return population
    
    def _update_alpha_beta_delta(self, fitness_values: np.ndarray):
        """Update alpha, beta, and delta wolves based on fitness"""
        sorted_indices = np.argsort(fitness_values)
        
        # Alpha (best solution)
        if fitness_values[sorted_indices[0]] < self.alpha_score:
            self.alpha_score = fitness_values[sorted_indices[0]]
            self.alpha_pos = self.positions[sorted_indices[0]].copy()
            
        # Beta (second best)
        if fitness_values[sorted_indices[1]] < self.beta_score:
            self.beta_score = fitness_values[sorted_indices[1]]
            self.beta_pos = self.positions[sorted_indices[1]].copy()
            
        # Delta (third best)
        if fitness_values[sorted_indices[2]] < self.delta_score:
            self.delta_score = fitness_values[sorted_indices[2]]
            self.delta_pos = self.positions[sorted_indices[2]].copy()
    
    def _update_positions(self, iteration: int):
        """Update wolf positions based on alpha, beta, delta"""
        a = 2 - iteration * (2 / self.max_iter)  # Linearly decreasing from 2 to 0
        
        for i in range(self.n_wolves):
            for j in range(self.dim):
                # Update position based on alpha
                r1, r2 = np.random.random(2)
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = abs(C1 * self.alpha_pos[j] - self.positions[i, j])
                X1 = self.alpha_pos[j] - A1 * D_alpha
                
                # Update position based on beta
                r1, r2 = np.random.random(2)
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = abs(C2 * self.beta_pos[j] - self.positions[i, j])
                X2 = self.beta_pos[j] - A2 * D_beta
                
                # Update position based on delta
                r1, r2 = np.random.random(2)
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = abs(C3 * self.delta_pos[j] - self.positions[i, j])
                X3 = self.delta_pos[j] - A3 * D_delta
                
                # Final position update
                self.positions[i, j] = (X1 + X2 + X3) / 3
                
                # Boundary checking
                self.positions[i, j] = np.clip(
                    self.positions[i, j], 
                    self.bounds[j, 0], 
                    self.bounds[j, 1]
                )
    
    def optimize(self,args, fitness_function, verbose: bool = True):
        """
        Run the optimization process
        
        Args:
            fitness_function: Function to evaluate fitness of hyperparameters
            verbose: Whether to print progress
            
        Returns:
            best_params: Best hyperparameters found
            best_score: Best fitness score achieved
        """
        print("Starting Grey Wolf Optimization...")
        start_time = time.time()
        
        for iteration in range(self.max_iter):
            # Evaluate fitness for all wolves
            for i in range(self.n_wolves):
                self.fitness_values[i] = fitness_function(args, self.positions[i],i)
            
            # Update alpha, beta, delta
            self._update_alpha_beta_delta(self.fitness_values)
            
            # Update positions
            self._update_positions(iteration)
            
            # Track convergence
            self.convergence_curve.append(self.alpha_score)
            
            if verbose:
                elapsed_time = time.time() - start_time
                print(f"Iteration {iteration + 1}/{self.max_iter} - "
                      f"Best Score: {self.alpha_score:.6f} - "
                      f"Time: {elapsed_time:.2f}s")
        
        total_time = time.time() - start_time
        print(f"Optimization completed in {total_time:.2f} seconds")
        
        return self.alpha_pos, self.alpha_score
    
    def plot_convergence(self):
        """Plot convergence curve"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.convergence_curve, linewidth=2)
        plt.title('GWO Convergence Curve')
        plt.xlabel('Iteration')
        plt.ylabel('Fitness Value')
        plt.grid(True, alpha=0.3)
        plt.show()
class ParticleSwarmOptimizer:
    """
    Particle Swarm Optimization for hyperparameter tuning.
    """
    def __init__(self, 
                 dim: int,
                 bounds: List[Tuple[float, float]],
                 n_particles: int = 30,
                 max_iter: int = 100,
                 w: float = 0.5,           # inertia
                 c1: float = 1.5,          # cognitive
                 c2: float = 1.5,          # social
                 random_seed: int = None):
        self.dim = dim
        self.bounds = np.array(bounds)
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2

        if random_seed:
            np.random.seed(random_seed)

        self.positions = self._initialize_population()
        self.velocities = np.zeros((n_particles, dim))
        self.personal_best_positions = self.positions.copy()
        self.personal_best_scores = np.full(n_particles, float('inf'))

        self.global_best_position = np.zeros(dim)
        self.global_best_score = float('inf')

        self.convergence_curve = []

    def _initialize_population(self) -> np.ndarray:
        population = np.zeros((self.n_particles, self.dim))
        for i in range(self.n_particles):
            for j in range(self.dim):
                population[i, j] = np.random.uniform(
                    self.bounds[j, 0], self.bounds[j, 1]
                )
        return population

    def optimize(self,args, fitness_function, verbose=True):
        print("Starting Particle Swarm Optimization...")
        start_time = time.time()

        for iteration in range(self.max_iter):
            for i in range(self.n_particles):
                score = fitness_function(args, self.positions[i], iteration)

                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = self.positions[i].copy()

                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.positions[i].copy()

            for i in range(self.n_particles):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                cognitive = self.c1 * r1 * (self.personal_best_positions[i] - self.positions[i])
                social = self.c2 * r2 * (self.global_best_position - self.positions[i])
                self.velocities[i] = self.w * self.velocities[i] + cognitive + social
                self.positions[i] += self.velocities[i]

                # Clip to bounds
                self.positions[i] = np.clip(self.positions[i], self.bounds[:, 0], self.bounds[:, 1])

            self.convergence_curve.append(self.global_best_score)
            if verbose:
                print(f"Iteration {iteration + 1}/{self.max_iter} - "
                      f"Best Score: {self.global_best_score:.6f} - "
                      f"Time: {time.time() - start_time:.2f}s")

        print(f"Optimization completed in {time.time() - start_time:.2f} seconds")
        return self.global_best_position, self.global_best_score

    def plot_convergence(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.convergence_curve, linewidth=2)
        plt.title('PSO Convergence Curve')
        plt.xlabel('Iteration')
        plt.ylabel('Fitness Value')
        plt.grid(True, alpha=0.3)
        plt.show()


class HyperparameterMapper:
    """Maps continuous GWO outputs to discrete/categorical hyperparameters"""
    
    def __init__(self):
        self.param_config = {
            'learning_rate': {'type': 'log', 'bounds': (1e-5, 1e-1)},
            #'embedding_dim': {'type': 'discrete', 'choices': [16, 32, 64, 128]},
            'num_transformer_layers': {'type': 'discrete', 'choices': [1, 2, 3, 4]},
            'num_attention_heads': {'type': 'discrete', 'choices': [2, 4, 8, 16]},
            'dropout_rate': {'type': 'continuous', 'bounds': (0.0, 0.5)},
            'bilstm_hidden_size': {'type': 'discrete', 'choices': [16, 32, 64, 128]},
            #'batch_size': {'type': 'discrete', 'choices': [8, 16, 32, 64]},
            'weight_decay': {'type': 'log', 'bounds': (1e-6, 1e-2)},
            'clip_grad_norm': {'type': 'continuous', 'bounds': (0.1, 5.0)},
            'scheduler_gamma': {'type': 'continuous', 'bounds': (0.1, 0.9)}
        }
    
    def map_params(self, gwo_output: np.ndarray) -> Dict[str, Any]:
        """Map GWO continuous output to actual hyperparameters"""
        mapped_params = {}
        
        for i, (param_name, config) in enumerate(self.param_config.items()):
            value = gwo_output[i]
            
            if config['type'] == 'log':
                # Logarithmic scaling
                log_min = np.log10(config['bounds'][0])
                log_max = np.log10(config['bounds'][1])
                mapped_value = 10 ** (log_min + value * (log_max - log_min))
                mapped_params[param_name] = mapped_value
                
            elif config['type'] == 'discrete':
                # Map to discrete choices
                idx = int(value * len(config['choices']))
                idx = min(idx, len(config['choices']) - 1)  # Ensure within bounds
                mapped_params[param_name] = config['choices'][idx]
                
            elif config['type'] == 'continuous':
                # Linear scaling
                min_val, max_val = config['bounds']
                mapped_value = min_val + value * (max_val - min_val)
                mapped_params[param_name] = mapped_value
        
        return mapped_params
    
    def get_bounds(self) -> List[Tuple[float, float]]:
        """Get bounds for GWO (all parameters normalized to [0, 1])"""
        return [(0.0, 1.0) for _ in self.param_config]


class STARModelTrainer:
    """Trainer for STAR model with hyperparameter optimization"""
    
    def __init__(self, 
                 model_class,
                 dataloader,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize trainer
        
        Args:
            model_class: STAR model class
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            device: Device to train on
        """
        self.model_class = model_class
        self.dataloader = dataloader
        self.device = device
        self.mapper = HyperparameterMapper()
        
    def create_model(self, params: Dict[str, Any], args):
        """Create model with given hyperparameters"""

        model = self.model_class(args, dropout_prob=params['dropout_rate'], nlayers = params['num_transformer_layers'], nhead = params['num_attention_heads'], bilstm_hidden_size = params['bilstm_hidden_size'])
        return model
    
    def train_and_evaluate(self,args, params: Dict[str, Any], max_epochs: int = 30) -> float:
        """
        Train model with given hyperparameters and return validation loss
        
        Args:
            params: Hyperparameters dictionary
            max_epochs: Maximum training epochs
            
        Returns:
            validation_loss: Final validation loss (fitness score)
        """
        try:

            args.num_transformer_layers = params['num_transformer_layers']
            args.num_attention_heads = params['num_attention_heads']
            args.dropout_rate = params['dropout_rate']
            args.bilstm_hidden_size = params['bilstm_hidden_size']
            self.using_cuda = torch.cuda.is_available()
        
            # Create model
            model = self.create_model(params, args)
            model = model.to(self.device)
            
            # Create optimizer and scheduler
            optimizer = Adam(
                model.parameters(),
                lr=params['learning_rate'],
                weight_decay=params['weight_decay']
            )
            
            milestones = [max_epochs // 2]  # Adjust based on your needs
            scheduler = MultiStepLR(
                optimizer, 
                milestones=milestones, 
                gamma=params['scheduler_gamma']
            )
            
            criterion = nn.MSELoss(reduction='none')
            
            # Training loop
            model.train()
            best_val_loss = float('inf')
            patience = 10
            patience_counter = 0
            
            for epoch in range(max_epochs):
                total_train_loss = 0
                num_batches = 0
                self.dataloader.reset_batch_pointer(set='train', valid=False)
                for batch in range(self.dataloader.trainbatchnums):                    
                    if batch >= 5:  # Limit batches for faster evaluation
                        break
                    inputs, batch_id = self.dataloader.get_train_batch(batch)    
                    inputs = tuple([torch.Tensor(i).to(self.device) for i in inputs])
                    
                    batch_abs, batch_norm, shift_value, seq_list, nei_list, nei_num, batch_pednum = inputs
                    inputs_forward = (
                        batch_abs[:-1], batch_norm[:-1], shift_value[:-1], 
                        seq_list[:-1], nei_list[:-1], nei_num[:-1], 
                        batch_pednum
                    )
                    
                    optimizer.zero_grad()
                    outputs = model.forward(inputs_forward, iftest=False)
                    
                    # Calculate loss (simplified version)
                    loss = criterion(outputs, batch_norm[1:, :, :3]).mean()
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 
                        params['clip_grad_norm']
                    )
                    optimizer.step()
                    
                    total_train_loss += loss.item()
                    num_batches += 1
                
                # Validation
                model.eval()
                total_val_loss = 0
                val_batches = 0
                
                with torch.no_grad():
                    self.dataloader.reset_batch_pointer(set='test')
                    for batch in range(self.dataloader.testbatchnums):                    
                        inputs, batch_id = self.dataloader.get_test_batch(batch)    
                        if batch >= 3:  # Limit validation batches
                            break
                            
                        inputs = tuple([torch.Tensor(i).to(self.device) for i in inputs])
                        batch_abs, batch_norm, shift_value, seq_list, nei_list, nei_num, batch_pednum = inputs
                        inputs_forward = (
                            batch_abs[:-1], batch_norm[:-1], shift_value[:-1],
                            seq_list[:-1], nei_list[:-1], nei_num[:-1],
                            batch_pednum
                        )
                        
                        outputs = model.forward(inputs_forward, iftest=True)
                        loss = criterion(outputs, batch_norm[1:, :, :3]).mean()
                        
                        total_val_loss += loss.item()
                        val_batches += 1
                
                avg_val_loss = total_val_loss / max(val_batches, 1)
                
                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break
                
                scheduler.step()
                model.train()
            
            return best_val_loss
            
        except Exception as e:
            print(f"Error during training: {e}")
            traceback.print_exc()
            return float('inf')  # Return worst possible score
    
    def fitness_function(self,args, gwo_output: np.ndarray, itteration) -> float:
        """
        Fitness function for GWO
        
        Args:
            gwo_output: Normalized hyperparameters from GWO
            
        Returns:
            fitness_score: Lower is better (validation loss)
        """
        # Map GWO output to actual hyperparameters
        params = self.mapper.map_params(gwo_output)
        
        print(f"{itteration} Evaluating hyperparameters: {params}")
        
        # Train and evaluate model
        val_loss = self.train_and_evaluate(args, params, max_epochs=30)  # Reduced epochs for faster evaluation
        
        print(f"Validation loss: {val_loss:.6f}")
        
        return val_loss


def run_hyperparameter_optimization(model_class, dataloader, args):
    """
    Main function to run hyperparameter optimization
    
    Args:
        model_class: STAR model class
        train_dataloader: Training data loader
        val_dataloader: Validation data loader
        
    Returns:
        best_params: Best hyperparameters found
        best_score: Best validation loss achieved
    """
    # Initialize trainer
    trainer = STARModelTrainer(model_class, dataloader)
    
    # Get bounds for optimization
    bounds = trainer.mapper.get_bounds()
    
    # Initialize GWO
    #optimizer = GreyWolfOptimizer(
    #    dim=len(bounds),
    #    bounds=bounds,
    #    n_wolves=30,  # Reduced for faster convergence
    #    max_iter=30,  # Reduced for demonstration
    #    random_seed=42
    #)
    optimizer = ParticleSwarmOptimizer(
        dim=len(bounds),
        bounds=bounds,
        n_particles=30,
        max_iter=10,
        random_seed=42
    )
    # Run optimization
    best_output, best_score = optimizer.optimize(args, trainer.fitness_function, verbose=True)
    
    # Map best parameters
    best_params = trainer.mapper.map_params(best_output)
    
    # Plot convergence
    optimizer.plot_convergence()
    
    print("\n" + "="*50)
    print("OPTIMIZATION RESULTS")
    print("="*50)
    print(f"Best validation loss: {best_score:.6f}")
    print("\nBest hyperparameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    return best_params, best_score


# Example usage
def StartTest(args):
    # Assuming you have your STAR model class and data loaders ready
    # from your_model_file import STAR
    # from your_data_file import get_dataloaders
    dataloader = Trajectory_Dataloader(args)    
    best_hyperparams, best_loss = run_hyperparameter_optimization(LSTM_STAR, dataloader, args)
    
    print("Hyperparameter optimization code is ready!")
    print("Please uncomment the example usage section and provide your model class and data loaders.")
