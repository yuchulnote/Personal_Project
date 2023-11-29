import numpy as np

class BatchNormalization:
    
    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.input_shape = None
        
        self.running_mean = running_mean
        self.running_var = running_var
        
        self.batch_size = None
        self.xc = None
        self.std = None
        self.dgamma = None
        self.dbeta = None
    
        
    def forward(self, x, train_flag=True):
        self.input_shape = x.shape
        if x.ndim != 2:
            Number, Channel, Height, Width = x.shape
            x = x.reshape(Number, -1)
            
        out = self.__forward(x, train_flag)
        
        return out.reshape(*self.input_shape)
    
    
    def __forward(self, x, train_flag):
        if self.running_mean is None:
            Number, Dimension = x.shape
            self.running_mean = np.zeros(Dimension)
            self.running_var = np.zeros(Dimension)
            
        if train_flag:
            mu = x.mean(axis=0)
            centered_input = x - mu
            var = np.mean(centered_input**2, axis=0)
            std = np.sqrt(var + 10e-7)
            normalized_input = centered_input / std
            
            self.batch_size = x.shape[0]
            self.centered_input = centered_input
            self.normalized_input = normalized_input
            self.std = std
            self.running_mean = self.momentum * self.running_mean + (1-self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1-self.momentum) * var
        else:
            centered_input = x - self.running_mean
            normalized_input = centered_input / ((np.sqrt(self.running_var + 10e-7)))
            
        out = self.gamma * normalized_input + self.beta
        
        return out
    
            