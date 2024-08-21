import torch
from torch import nn 

def trainer(
    model: nn.Module, 
    X_train_set,
    y_train_set,
    X_val_set,
    y_val_set,
    epochs,
    weight_decay=0,
    lr = 1e-4,
    device="cpu"
):
    
    model = model.to(device)
    model.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_func = nn.CrossEntropyLoss()
    
    print("Training starts")
    for epoch in range(epochs):
        loss_hitory = []
        for x_batch, y_batch in zip(X_train_set, y_train_set):
            x_batch = x_batch.to(device) 
            y_batch = y_batch.to(device) 

            optimizer.zero_grad()
            
            # print(x_batch.shape)
            pred = model(x_batch)
            loss = loss_func(pred, y_batch)
            # print(pred.shape, y_batch.shape)
            # print("loss", loss.item())
            loss_hitory.append(loss.item())
            loss.backward()
            optimizer.step()
        avg_loss = sum(loss_hitory) / len(loss_hitory)
        print(f"Average Loss: {avg_loss}")
