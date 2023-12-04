import numpy as np
import os

class EarlyStopping:
    def __init__(self, patience=None, verbose=False, delta=0, base_path='Checkpoint', save_path='./'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.lnf
        self.delta = delta
        self.base_path = base_path
        self.save_path = save_path
    
    
    def save_checkpoint(self, val_loss, model, optimizer, scheduler, scaler, epoch):
        checkpoint_path = os.path.join(self.save_path, f"{self.base_path}_Best_{epoch+1}.pt")
        checkpoint = {
            'model_state_dict' : model.state_dict(),
            'optimizer_state_dict' : optimizer.state_dict(),
            'scheduler_state_dict' : scheduler.state_dict(),
            'scaler_state_dict' : scaler.state_dict(),
            'epoch' : epoch,
            'val_loss' : val_loss
        }
        if self.verbose:
            print(f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model to {checkpoint} ...")
        torch.save(checkpoint, checkpoint_path)
        self.val_loss_min = val_loss
        
    def step(self, val_loss, model, optimizer, scheduler, scaler, epoch):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, scheduler, scaler, epoch)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping Count : {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, scheduler, scaler, epoch)
            self.counter = 0
    
            