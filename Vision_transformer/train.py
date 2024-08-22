import torch
from torch import nn 

def calc_accuracy(images, lables, model, device):
    model.eval()

    total = 0
    n_corrects = 0
    with torch.no_grad():
        for x_batch, y_batch in zip(images, lables):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            N = x_batch.shape[0]
            
            pred = model(x_batch)
            n_corrects += pred.argmax(dim=1).eq(y_batch).sum().item()
            total += N
    return n_corrects / total

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
    
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_func = nn.CrossEntropyLoss()
    
    training_acc_history = []
    val_acc_history = []
    
    training_loss_history = []
    
    best_val_accuracy = 0
    best_params = None
    stop_cnt = 5
    
    iterations_per_epoch = len(X_train_set)
    
    print("Training starts")
    for epoch in range(epochs):
        model.train()
        for x_batch, y_batch in zip(X_train_set, y_train_set):
            x_batch = x_batch.to(device) 
            y_batch = y_batch.to(device) 

            optimizer.zero_grad()
            
            pred = model(x_batch)
            loss = loss_func(pred, y_batch)

            training_loss_history.append(loss.item())
            loss.backward()
            optimizer.step()
        
        curr_loss_history = training_loss_history[iterations_per_epoch*epoch:iterations_per_epoch*(epoch+1)]
        avg_loss = sum(curr_loss_history) / len(curr_loss_history)
        
        val_accuracy = calc_accuracy(X_val_set, y_val_set, model, device)
        training_accuracy = calc_accuracy(X_train_set, y_train_set, model, device)
        
        training_acc_history.append(training_accuracy)
        val_acc_history.append(val_accuracy)
        
        print(f"Epoch: {epoch+1} Average Loss: {avg_loss} Validation Accuracy: {val_accuracy}" )
        
        if val_acc_history > best_val_accuracy:
            best_val_accuracy = val_acc_history
            best_params = model.state_dict()
            stop_cnt = 0
        else:
            stop_cnt += 1
        
        if stop_cnt >= early_stop:
            break
        
    return training_loss_history, training_acc_history, val_acc_history, best_params

