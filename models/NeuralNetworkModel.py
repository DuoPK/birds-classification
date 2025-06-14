import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils.ClassificationMetrics import ClassificationMetrics

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate=0.2):
        super(NeuralNetwork, self).__init__()
        layers = []
        
        # Input layer
        prev_size = input_size
        
        # Hidden layers
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class NeuralNetworkModel:
    def __init__(self, input_size, output_size=2, 
                 learning_rate=0.001, batch_size=32, epochs=100, 
                 device=None, **kwargs):
        """
        Parameters:
        - input_size: number of input features
        - output_size: number of output classes
        - learning_rate: learning rate for optimizer
        - batch_size: batch size for training
        - epochs: number of training epochs
        - device: device to run the model on ('cuda' or 'cpu')
        - **kwargs: additional parameters including:
            - n_layers: number of hidden layers (from Optuna)
            - layer_X_size: size of layer X (from Optuna)
            - dropout_rate: dropout rate for layers
        """
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        
        # Construct hidden_sizes from Optuna parameters
        if 'hidden_sizes' in kwargs:
            hidden_sizes = kwargs.pop('hidden_sizes')
        else:
            hidden_sizes = []
            n_layers = kwargs.pop('n_layers', 2)  # Default to 2 layers if not specified
            for i in range(n_layers):
                layer_size = kwargs.pop(f'layer_{i}_size', 64)  # Default to 64 if not specified
                hidden_sizes.append(layer_size)
        
        self.hidden_sizes = hidden_sizes
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        self.model = NeuralNetwork(input_size, hidden_sizes, output_size, **kwargs)
        
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
    def train(self, X_train, y_train, epochs=None):
        if epochs is None:
            epochs = self.epochs
            
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.LongTensor(y_train).to(self.device)
        
        self.model.train()
        for epoch in range(epochs):
            # Create batches
            indices = torch.randperm(len(X_train))
            for i in range(0, len(X_train), self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                X_batch = X_train[batch_indices]
                y_batch = y_train[batch_indices]
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
    
    def predict(self, X_test):
        self.model.eval()
        with torch.no_grad():
            X_test = torch.FloatTensor(X_test).to(self.device)
            outputs = self.model(X_test)
            _, predicted = torch.max(outputs.data, 1)
            return predicted.cpu().numpy()
    
    def evaluate(self, X_test, y_test, positive_label=1):
        y_pred = self.predict(X_test)
        metrics = ClassificationMetrics(y_test, y_pred, positive_label=positive_label)
        return metrics.summary()
    
    def get_params(self):
        return {
            'input_size': self.input_size,
            'hidden_sizes': self.hidden_sizes,
            'output_size': self.output_size,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'device': str(self.device)
        }

    def get_model_structure(self):
        """Returns a string representation of the model structure"""
        structure = f"Neural Network Structure:\n"
        structure += f"Input Layer: {self.input_size} features\n"
        for i, size in enumerate(self.hidden_sizes):
            structure += f"Hidden Layer {i+1}: {size} neurons\n"
        structure += f"Output Layer: {self.output_size} neurons"
        return structure 