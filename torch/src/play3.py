import torch
import torchvision
from torch import nn, optim
from data_augmentation.load_data import data_preprocess 


device = "cpu"
dtype = torch.float32
n_samples = 100
    
x_train, y_train, x_valids, y_valids, X_test, y_test = data_preprocess(image_show=False, n_samples=n_samples, validation_ratio=0.2, dtype=dtype)


model = torchvision.models.resnet34(pretrained=False)

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 15
batch_size = 10
print_every = 20

N = x_train.shape[0]
n_batches_per_iteration = N // batch_size
n_iterations = epochs * n_batches_per_iteration 

def evaluate_model(model, X, y, device, batch_size):
    model.eval()
    
    N = X.shape[0]
    X = X.to(device)
    y = y.to(device)
    
    n_batches = N // batch_size
    
    scores = []
    for k in range(n_batches):
        s = k * batch_size
        e = k * batch_size + batch_size
        sub_x = X[s:e]
        s = model(sub_x)
        # print("scores", scores)
        scores.append(s.argmax(dim=1))
        
    scores = torch.cat(scores)
    result = (scores == y).to(dtype=torch.float).mean().item()
    return result

print(n_iterations)
for t in range(n_iterations):
    model.train()  # Set the model to training mode
    N = x_train.shape[0]
    batch_mask = torch.randperm(N)[:batch_size]
    batch_x = x_train[batch_mask].to(device)
    batch_y = y_train[batch_mask].to(device)
        
    optimizer.zero_grad()  # Zero the parameter gradients

    outputs = model(batch_x)
    loss = criterion(outputs, batch_y)
    loss.backward()
    optimizer.step()

    end_epoch = (t + 1) % n_batches_per_iteration == 0
    
    if t % print_every == 0:
        print(f"Iteration {t+1}/{n_iterations}: {loss.item()}")
        
    with torch.no_grad():
        start = t == 0
        end = t == n_iterations - 1
        if start or end or end_epoch:
            train_acc = evaluate_model(model, x_train, y_train, device, batch_size)
            val_accuracy = evaluate_model(model, x_valids, y_valids, device, batch_size)
            print("train accuracy:", train_acc)
            print("val_accuracy:", val_accuracy)

