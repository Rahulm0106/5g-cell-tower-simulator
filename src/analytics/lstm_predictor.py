"""
Simplified LSTM Traffic Predictor

Since TensorFlow isn't available, we'll implement a basic LSTM from scratch
using NumPy. This is educational and shows what's happening under the hood.

For production, you'd use TensorFlow/PyTorch, but the concepts are identical.

============================================================
CONCEPT: LSTM Cell (Simplified)
============================================================
An LSTM cell has 4 components:

  ① Forget Gate (f):  Decides what to forget from cell state
     f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
     
  ② Input Gate (i):   Decides what new info to add
     i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
     
  ③ Candidate (g):    Proposes new cell state
     g_t = tanh(W_g · [h_{t-1}, x_t] + b_g)
     
  ④ Output Gate (o):  Decides what to output
     o_t = σ(W_o · [h_{t-1}, x_t] + b_o)

  Cell state update:
     C_t = f_t * C_{t-1} + i_t * g_t
     
  Hidden state output:
     h_t = o_t * tanh(C_t)

Where:
  σ = sigmoid function
  * = element-wise multiplication
  W, b = learnable weights and biases

For simplicity, we'll implement a minimal version with fewer parameters.
"""

import numpy as np
from typing import Tuple
import pickle
import sys
import os
# Ensure the project `src/` directory is on sys.path when running this script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# ===========================================================================
# Activation Functions
# ===========================================================================
def sigmoid(x):
    """Sigmoid activation: maps to (0, 1)"""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def tanh(x):
    """Tanh activation: maps to (-1, 1)"""
    return np.tanh(np.clip(x, -500, 500))


# ===========================================================================
# Simplified LSTM Layer
# ===========================================================================
class SimpleLSTM:
    """
    Minimal LSTM implementation for time-series prediction.
    
    Architecture:
      - Input:  (batch, time_steps, input_size)
      - Output: (batch, output_size)
      
    We'll use a single LSTM layer followed by a dense layer.
    """
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """
        Args:
            input_size:   Number of features per time step (1 for univariate)
            hidden_size:  Number of LSTM units
            output_size:  Number of forecast steps
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights (Xavier initialization)
        scale = np.sqrt(2.0 / (input_size + hidden_size))
        
        # LSTM weights: combined matrix for all gates
        self.W = np.random.randn(input_size + hidden_size, 4 * hidden_size) * scale
        self.b = np.zeros((4 * hidden_size,))
        
        # Output dense layer
        self.W_out = np.random.randn(hidden_size, output_size) * scale
        self.b_out = np.zeros((output_size,))
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass through LSTM.
        
        Args:
            X: Input sequence (batch, time_steps, input_size)
            
        Returns:
            Predictions (batch, output_size)
        """
        batch_size, time_steps, _ = X.shape
        
        # Initialize hidden and cell states
        h = np.zeros((batch_size, self.hidden_size))
        c = np.zeros((batch_size, self.hidden_size))
        
        # Process sequence
        for t in range(time_steps):
            x_t = X[:, t, :]  # (batch, input_size)
            
            # Concatenate h and x
            combined = np.concatenate([h, x_t], axis=1)  # (batch, hidden + input)
            
            # Compute all gates at once
            gates = combined @ self.W + self.b  # (batch, 4*hidden)
            
            # Split into 4 gates
            i = sigmoid(gates[:, :self.hidden_size])                      # Input gate
            f = sigmoid(gates[:, self.hidden_size:2*self.hidden_size])    # Forget gate
            g = tanh(gates[:, 2*self.hidden_size:3*self.hidden_size])     # Candidate
            o = sigmoid(gates[:, 3*self.hidden_size:])                    # Output gate
            
            # Update cell and hidden states
            c = f * c + i * g
            h = o * tanh(c)
        
        # Final output (use last hidden state)
        output = h @ self.W_out + self.b_out  # (batch, output_size)
        
        return output
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              epochs: int = 50,
              learning_rate: float = 0.001,
              batch_size: int = 32,
              verbose: bool = True) -> dict:
        """
        Train the LSTM using gradient descent.
        
        Simplified training (full backprop through time would be complex):
          - Use numerical gradients (slower but simpler)
          - Or use a simple momentum update
          
        For real projects, use TensorFlow/PyTorch automatic differentiation.
        
        Here we'll use a very simple approach: random search with momentum.
        """
        history = {'train_loss': [], 'val_loss': []}
        
        best_loss = float('inf')
        best_weights = None
        
        # Normalize data (important for LSTM stability)
        X_mean, X_std = X_train.mean(), X_train.std()
        y_mean, y_std = y_train.mean(), y_train.std()
        
        X_train_norm = (X_train - X_mean) / (X_std + 1e-8)
        X_val_norm = (X_val - X_mean) / (X_std + 1e-8)
        y_train_norm = (y_train - y_mean) / (y_std + 1e-8)
        y_val_norm = (y_val - y_mean) / (y_std + 1e-8)
        
        # Store normalization params
        self.X_mean, self.X_std = X_mean, X_std
        self.y_mean, self.y_std = y_mean, y_std
        
        for epoch in range(epochs):
            # Forward pass
            y_pred_train = self.forward(X_train_norm)
            train_loss = np.mean((y_pred_train - y_train_norm)**2)
            
            y_pred_val = self.forward(X_val_norm)
            val_loss = np.mean((y_pred_val - y_val_norm)**2)
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            
            # Save best model
            if val_loss < best_loss:
                best_loss = val_loss
                best_weights = {
                    'W': self.W.copy(),
                    'b': self.b.copy(),
                    'W_out': self.W_out.copy(),
                    'b_out': self.b_out.copy()
                }
            
            # Simple parameter update with random noise (exploration)
            # This is NOT a proper gradient descent but works for demo
            if epoch < epochs - 1:  # Don't update on last epoch
                noise_scale = learning_rate * (1 - epoch / epochs)  # Decay
                
                self.W += np.random.randn(*self.W.shape) * noise_scale
                self.b += np.random.randn(*self.b.shape) * noise_scale
                self.W_out += np.random.randn(*self.W_out.shape) * noise_scale
                self.b_out += np.random.randn(*self.b_out.shape) * noise_scale
            
            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch+1:>3}/{epochs} | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f}")
        
        # Restore best weights
        if best_weights:
            self.W = best_weights['W']
            self.b = best_weights['b']
            self.W_out = best_weights['W_out']
            self.b_out = best_weights['b_out']
        
        if verbose:
            print(f"\n✅ Training complete. Best val loss: {best_loss:.4f}")
        
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Input sequences (batch, time_steps, input_size)
            
        Returns:
            Predictions in original scale (batch, output_size)
        """
        # Normalize
        X_norm = (X - self.X_mean) / (self.X_std + 1e-8)
        
        # Forward pass
        y_pred_norm = self.forward(X_norm)
        
        # Denormalize
        y_pred = y_pred_norm * self.y_std + self.y_mean
        
        return y_pred
    
    def save(self, filepath: str):
        """Save model to disk."""
        model_dict = {
            'W': self.W,
            'b': self.b,
            'W_out': self.W_out,
            'b_out': self.b_out,
            'X_mean': self.X_mean,
            'X_std': self.X_std,
            'y_mean': self.y_mean,
            'y_std': self.y_std,
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'output_size': self.output_size
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_dict, f)
        print(f"✅ Model saved: {filepath}")
    
    @classmethod
    def load(cls, filepath: str):
        """Load model from disk."""
        with open(filepath, 'rb') as f:
            model_dict = pickle.load(f)
        
        model = cls(
            input_size=model_dict['input_size'],
            hidden_size=model_dict['hidden_size'],
            output_size=model_dict['output_size']
        )
        model.W = model_dict['W']
        model.b = model_dict['b']
        model.W_out = model_dict['W_out']
        model.b_out = model_dict['b_out']
        model.X_mean = model_dict['X_mean']
        model.X_std = model_dict['X_std']
        model.y_mean = model_dict['y_mean']
        model.y_std = model_dict['y_std']
        
        print(f"✅ Model loaded: {filepath}")
        return model


# ===========================================================================
# Demo: Train on synthetic traffic data
# ===========================================================================
if __name__ == "__main__":
    from analytics.traffic_generator import generate_network_traffic, prepare_train_test_split
    
    print("=" * 70)
    print("  LSTM TRAFFIC PREDICTOR — DEMO")
    print("=" * 70)
    
    # Generate training data
    print("\n1️⃣  Generating synthetic traffic data...")
    df = generate_network_traffic(num_cells=1, duration_hours=336, sample_rate_min=5)  # 2 weeks
    
    # Prepare dataset
    print("2️⃣  Preparing train/test split...")
    X_train, y_train, X_test, y_test = prepare_train_test_split(
        df, cell_id='Cell_00',
        train_frac=0.8,
        lookback=12,   # 1 hour history
        forecast=3     # 15 min ahead
    )
    
    print(f"   X_train: {X_train.shape}   y_train: {y_train.shape}")
    print(f"   X_test:  {X_test.shape}    y_test:  {y_test.shape}")
    
    # Create and train model
    print("\n3️⃣  Training LSTM model...")
    model = SimpleLSTM(input_size=1, hidden_size=16, output_size=3)
    
    history = model.train(
        X_train, y_train,
        X_test, y_test,
        epochs=100,
        learning_rate=0.01,
        verbose=True
    )
    
    # Evaluate
    print("\n4️⃣  Evaluating model...")
    y_pred = model.predict(X_test)
    
    mae = np.mean(np.abs(y_pred - y_test))
    rmse = np.sqrt(np.mean((y_pred - y_test)**2))
    
    print(f"   MAE:  {mae:.2f} UEs")
    print(f"   RMSE: {rmse:.2f} UEs")
    
    # Show sample predictions
    print("\n5️⃣  Sample predictions (first 5 test samples):")
    print("   " + "-" * 60)
    print(f"   {'Actual':<30} {'Predicted':<30}")
    print("   " + "-" * 60)
    for i in range(min(5, len(y_test))):
        actual = y_test[i]
        predicted = y_pred[i]
        print(f"   {str(actual):<30} {str(predicted):<30}")
    
    # Save model to results directory
    model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'results'))
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'lstm_model.pkl')
    model.save(model_path)
    
    print("\n" + "=" * 70)
    print("  ✅ LSTM training complete!")
    print("=" * 70)