import sys
sys.path.append("../")

import time
import torch
from src.model.DeepConv import DeepConvNet

class Solver2:
    """
        Solver class to run the model and save the key statistics: loss
    """
    def __init__(self, model: DeepConvNet, data, **kwargs):
        """
            initialization of Solver class
            
            Inputs:
                data: dictionary containing training, validation set
                model: our model to train the dataset
                kwargs:
                    epochs: num of training cycles
                    batch_size: minibatch size
                    update_rule: gradient update rule
                    optim_config: hyperparameters for each update rule. Requires learning rate to be set.
                    lr_decay: a scalar for learning rate decay. After each epoch the learning rate is multiplied by this amount.
                    n_training_samples: number of training samples for computing accuracy
                    n_val_samples: number of validation samples for computing accuracy
                    
                    device: "cpu" or "cuda" 
                    
        """
        self.X_train = data["X_train"]
        self.y_train = data["y_train"]
        self.X_val = data["X_val"]
        self.y_val = data["y_val"]

        self.model = model
        
        self.epochs = kwargs.pop("epochs", 10)
        self.batch_size = kwargs.pop("batch_size", 100)
        self.update_rule = kwargs.pop("update_rule", self.sgd)
        self.optim_config = kwargs.pop("optim_config", {})
        self.lr_decay = kwargs.pop("lr_decay", 1.0)
                
        self.n_train_samples = kwargs.pop("n_train_samples", 1000)
        self.n_val_samples = kwargs.pop("n_val_samples", None)
    
        self.device = kwargs.pop("device", "cpu")
        
        self.print_every = kwargs.pop("print_every", 20)
        
        self.reset()
        
    def reset(self):
        self.epoch = 0
        self.best_acc = 0
        self.best_params = {}
        self.loss_history = []
        self.training_acc_history = []
        self.val_acc_history = [] 

        self.optim_configs = {}
        for param in self.model.params:
            if isinstance(self.model.params[param], list):
                n = len(self.model.params[param])
                # just create a new config 
                # FIX THIS
                self.optim_configs[param] = [{k:v for k, v in self.optim_config.items()} for _ in range(n)] 
            else:
                config = {k:v for k, v in self.optim_config.items()}
                self.optim_configs[param] = config
    
    def sgd(self, w, dw, config):
        """
            Basic Stochastic gradient descent update rule
        """
        if config is None:
            config = {}

        config.setdefault("learning_rate", 1e-3)
        lr = config["learning_rate"]
        next_w = w - lr * dw
        return next_w, config
    
    
    def _step(self):
        N = self.X_train.shape[0]
        batch_mask = torch.randperm(N)[:self.batch_size]
        batch_x = self.X_train[batch_mask].to(self.device)
        batch_y = self.y_train[batch_mask].to(self.device)
        loss, grads = self.model.loss(batch_x, batch_y)
        
        self.loss_history.append(loss.item())
        print(self.model.params["W5"].shape)
        with torch.no_grad():
            for param in self.model.params:
                if isinstance(self.model.params[param], list):
                    # print("len", len(self.model.params[param]), len(grads[param]), len(self.optim_configs[param]))
                    # print(param)
                    for i, (w, dw, config) in enumerate(zip(self.model.params[param], grads[param], self.optim_configs[param])):
                        # print("w.shape, dw.shape", w.shape, dw.shape)
                        # print("running once")
                        next_w, next_config = self.update_rule(w, dw, config)
                        self.model.params[param][i] = next_w
                        self.optim_configs[param][i] = next_config
                else:
                    # print("running second")
                    # print(param)
                    # print(w.shape)
                    # print(config)
                    dw = grads[param]
                    config = self.optim_configs[param]
                    next_w, next_config = self.update_rule(w, dw, config)
                    self.model.params[param] = next_w
                    self.optim_configs[param] = next_config
                    
    def check_accuracy(self, X, y, num_samples=None, batch_size=100):
        """
            compute the accuracy of X compare to true label Y.
            
            if number of samples is not None, make a sub sample of data set.
            
            compute the accuracy in batches to avoid using too much memory
        """
        N = X.shape[0]
        if num_samples is not None:
            sub_mask = torch.randperm(N, device=self.device)[:num_samples]
            N = num_samples
            X = X[sub_mask]
            y = y[sub_mask]
        X = X.to(self.device)
        y = y.to(self.device)
        
        n_batches = N // batch_size
        
        if N % batch_size != 0:
            n_batches += 1

        scores = []
        for k in range(n_batches):
            s = k * batch_size
            e = k * batch_size + batch_size
            sub_x = X[s:e]
            s = self.model.loss(sub_x)
            scores.append(s.argmax(dim=1))

        scores = torch.cat(scores)
        result = (scores == y).to(dtype=torch.float).mean().item()
        return result
    
    def train(self):
        """
            training the model
        """
        N = self.X_train.shape[0]
        n_batches_per_iteration = N // self.batch_size
        n_iterations = self.epochs * n_batches_per_iteration 

        print("total number of iterations: ", n_iterations)
        starttime = time.time()
        for t in range(n_iterations):
            self._step()
            
            if t % self.print_every == 0:
                print(f"Iteration {t+1}/{n_iterations}: {self.loss_history[-1]}")
            
            end_epoch = (t + 1) % n_batches_per_iteration == 0
            
            if end_epoch:
                # later maybe try cosine 
                self.epoch += 1
                for p in self.optim_configs:
                    self.optim_configs[p]["learning_rate"] *= self.lr_decay
            
            with torch.no_grad():
                start = t == 0
                end = t == n_iterations - 1
                if start or end or end_epoch:
                    endtime = time.time()
                    print(f"time: {endtime - starttime}")
                    starttime = endtime
                    train_acc = self.check_accuracy(self.X_train, self.y_train)
                    val_acc = self.check_accuracy(self.X_val, self.y_val)
                    
                    self.training_acc_history.append(train_acc)
                    self.val_acc_history.append(val_acc)
                    
                    print(f"Epoch {self.epoch}/{self.epochs}: train acc: {train_acc:2f} validation acc: {val_acc:2f}")
                    
                
                    # if val_acc > self.best_acc:
                    #     self.best_acc = val_acc
                    #     self.best_params = {p : v.clone() for p,v in self.model.params.items()}
        
        # return best acc?