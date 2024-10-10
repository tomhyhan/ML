import torch
from torch import nn

def trainer(
    model: nn.Module, 
    X_train_set,
    y_train_set,
    X_val_set,
    y_val_set,
    epochs,
    weight_decay=1e-3,
    lr = 1e-4,
    early_stop=5,
    device="cpu"
):
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    loss_func = nn.CrossEntropyLoss()

    training_loss_history = []
    val_accuracy_history = []
    train_accuracy_history = []
    
    print("Start Training")
    model.train()
    for epoch in range(epochs):
        curr_losses = []
        for x_batch, y_batch in zip(X_train_set, y_train_set):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            optim.zero_grad()
            pred = model(x_batch)
            loss = loss_func(pred, y_batch)
            curr_losses.append(loss.item())
            loss.backward()
            optim.step()
            
        curr_avg_loss = sum(curr_losses) / len(curr_losses)
        training_loss_history.append(curr_avg_loss)

        validation_accuracy = calc_accuracy(model, X_val_set, y_val_set, device)
        train_accuracy = calc_accuracy(model, X_train_set, y_train_set, device)
        
        train_accuracy_history.append(train_accuracy)
        val_accuracy_history.append(validation_accuracy)
        
        print(f"Epoch {epoch + 1} Loss: {curr_avg_loss} Train Accuracy: {train_accuracy} Validation Accuracy: {validation_accuracy}")
        
        
def calc_accuracy(model, X, y, device="cpu"):
    model.eval()
    
    total = 0
    n_corrects = 0
    for x_batch, y_batch in zip(X,y):
        x_batch = x_batch.to(device)  
        y_batch = y_batch.to(device)
        pred = model(x_batch)
        
        N = x_batch.size(0)
        n_corrects += pred.argmax(dim=1).eq(y_batch).sum().item()
        total += N
    
    return n_corrects / total