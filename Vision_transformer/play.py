import os
import torch
from torch.utils.data import DataLoader
from torchvision.models import VisionTransformer

from data import preprocess_cifar10
from vtransformer import SimpleVisionTransformer
from train import trainer
from viz import draw_loss, draw_train_val_accuracy

data = preprocess_cifar10(n_test=100, n_train=200)

N, C, H, W = data["X_train"].shape

def load_checkpoint(model, checkpoint_path):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model parameters from {checkpoint_path}")
        return checkpoint
    else:
        print(f"No checkpoint found at {checkpoint_path}")
        return None
    
# model parameters
image_size = H
patch_size = 4
num_layers = 6
num_heads = 6
embedding_dim = 252
forward_dim = 252 * 4
dropout = 0.1
attention_dropout = 0.1
num_classes = 10
representation_size = 252 * 2

# train parameter
epochs = 5
batch_size = 32
lr = 1e-3
checkpoint_path = "./bestpath.pt"
device="cpu"

#  62 with representation
#  27 with representation + MLP
model = SimpleVisionTransformer(
    image_size,
    patch_size,
    num_layers,
    num_heads,
    embedding_dim,
    forward_dim,
    dropout,
    attention_dropout,
    num_classes,
    representation_size,
)

# 38 with representation
model = VisionTransformer(
    image_size=image_size,
    patch_size=patch_size,
    num_layers=num_layers,
    num_heads=num_heads,
    hidden_dim=embedding_dim,
    mlp_dim=forward_dim,
    dropout=dropout,
    attention_dropout=attention_dropout,
    num_classes=10,
    representation_size=representation_size,
    conv_stem_configs=None
)

X_train = data["X_train"]
y_train = data["y_train"]
X_val = data["X_val"]
y_val = data["y_val"]
X_test = data["X_test"]
y_test = data["y_test"]

X_train_set = DataLoader(X_train, batch_size=batch_size, shuffle=False, drop_last=True)
y_train_set = DataLoader(y_train, batch_size=batch_size, shuffle=False, drop_last=True)
X_val_set = DataLoader(X_val, batch_size=batch_size, shuffle=False, drop_last=True)
y_val_set = DataLoader(y_val, batch_size=batch_size, shuffle=False, drop_last=True)
X_test_set = DataLoader(X_test, batch_size=batch_size, shuffle=False, drop_last=True)
y_test_set = DataLoader(y_test, batch_size=batch_size, shuffle=False, drop_last=True)


training_loss_history, training_acc_history, val_accuracy_history, best_params = trainer(model, X_train_set, y_train_set, X_val_set, y_val_set, epochs)

# torch.save(best_params, checkpoint_path)

print(training_acc_history)
print(val_accuracy_history)
draw_loss(training_loss_history)
draw_train_val_accuracy(training_acc_history, val_accuracy_history)

# TODO
# 1. Qualitative examples
# 2. Confusion matrices
# 3. Attribution methods