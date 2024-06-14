import sys
sys.path.append("../")

import torch
from src.models.ResNet import ResNet
from src.layers.basicblock import BasicBlock

class Solver:
    """
        Solver class that runs the given model on the data input. 
    """
    def __init__(self, data, model: ResNet, **kwargs):
        """
            initialize and set all parameters needed for training.
            
            Inputs:
                data:
                    X_train: Input train data
                    y_train: True label for train data
                    X_val: Input validation data
                    y_val: True label for validation data
                model:
                    Model to train our data
                Params:
                    epochs: number of epochs
                    batch_size: batch_size
                    lr_decay: learning rate decay for each training (t)
                    update_rule: various kinds of SGD update algorithms. Default plain sgd.
                    device: device
                    dtype: dtype
                    
                    optim_config: params for optimization
                        learning_rate: default 0.001 
            
                    n_training_samples: number of training samples to check accuracy
                    n_val_samples: number of validation samples to check accuracy
                    print_every: print current time step and loss
        """
        self.X_train = data["X_train"]
        self.y_train = data["y_train"]
        self.X_val = data["X_val"]
        self.y_val = data["y_val"]
    
        self.model = model
        
        self.epochs = kwargs.pop("epochs", 5)
        self.batch_size = kwargs.pop("batch_size", 100)
        self.lr_decay = kwargs.pop("lr_decay", 0)
        self.update_rule = kwargs.pop("update_rule", self.sgd)
        self.device = kwargs.pop("device", "cpu")
        self.dtype = kwargs.pop("dtype", torch.float32)
        
        self.optim_config = kwargs.pop("optim_config", {
            "learning_rate": 0.001
        })
        
        self.n_training_samples = kwargs.pop("n_training_samples", 1000)
        self.n_val_samples = kwargs.pop("n_val_samples", None)
        self.print_every = kwargs.pop("print_every", 100)

        self.loss_history = []
        self.training_acc_history = []
        self.val_acc_history = []
        
        # for layer in self.model.param_layers:
        #  if isinstance(layer, BasicBlock):
            #  for param in layer.param_layers:
                #  for p in param.params:
        for layer in self.model.param_layers:
            if isinstance(layer, BasicBlock):
                for param in layer.param_layers:
                    for p in param.configs:
                        param.configs[p] = {k:v for k, v in self.optim_config.items()}
            else:
                for p in layer.params:
                    layer.configs[p] = {k:v for k, v in self.optim_config.items()}

    def sgd(self, w, dw, config):
        if len(config) == 1:
            config = {
                "learning_rate": 1e-3,
                "test": "test"
            }

        lr = config["learning_rate"]
        w -= lr * dw

        return config
    
    def _loss(self):
        """
            extract Random input data with the batch size the compute the loss. Then update the parameters
        """
        N = self.X_train.shape[0]
        mask = torch.randperm(N)[:self.batch_size]
        batch_x = self.X_train[mask].clone().to(self.device)
        batch_y = self.y_train[mask].clone().to(self.device)
        
        loss = self.model.loss(batch_x, batch_y)
        self.loss_history.append(loss)
        
        # print(self.model.param_layers[0].param_layers[0].w.shape)
        
        with torch.no_grad():
            for layer in self.model.param_layers:
                if isinstance(layer, BasicBlock):
                    for param in layer.param_layers:
                        for p in param.params:
                            w = param.params[p]
                            dw = param.grads[p]
                            config = param.configs[p]
                            next_config = self.update_rule(w,dw,config)
                            param.configs[p] = next_config
                        param.reset_grads()
                    #     break
                    # break
                else:
                    # fix: refactor this to another function
                    for p in layer.params:
                        w = layer.params[p]
                        dw = layer.grads[p]
                        config = layer.configs[p]
                        next_config = self.update_rule(w,dw,config)
                        layer.configs[p] = next_config
                    layer.reset_grads()
                # break
        
    def train(self):
        """
            train the data using the model.
        """
        
        N = self.X_train.shape[0]
        n_batch_iteration = N // self.batch_size
        n_iterations = n_batch_iteration * self.epochs
        
        for t in range(n_iterations):
            self._loss()
            if t % self.print_every == 0:
                print(f"train: {t} loss: {self.loss_history[-1]}", )
            # break
