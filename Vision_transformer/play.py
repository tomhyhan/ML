import torch
from torch.utils.data import DataLoader
from torchvision.models import VisionTransformer

from data import preprocess_cifar10
from vtransformer import SimpleVisionTransformer
from train import trainer

data = preprocess_cifar10(n_test=100, n_train=1000)

N, C, H, W = data["X_train"].shape

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
representation_size = None

# train parameter
epochs = 10
batch_size = 64
lr = 1e-3
device="cpu"

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

# model = VisionTransformer(
#     image_size=image_size,
#     patch_size=patch_size,
#     num_layers=num_layers,
#     num_heads=num_heads,
#     hidden_dim=embedding_dim,
#     mlp_dim=forward_dim,
#     dropout=dropout,
#     attention_dropout=attention_dropout,
#     num_classes=10,
#     representation_size=None,
#     conv_stem_configs=None
# )

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


trainer(model, X_train_set, y_train_set, X_val_set, y_val_set, epochs)