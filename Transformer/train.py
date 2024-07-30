import torch
from transformers import Transformers
from torch.optim.lr_scheduler import CosineAnnealingLR

def train_model(
    model,
    train_loader,
    val_loader,
    loss_func,
    lr,
    num_epochs,
    batch_size,
    device="cpu" 
):
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)
    
    iteration = 0
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = []
        for item in train_loader:
            que, que_pos, ans, ans_pos = item
            model.to(device)
            que = que.to(device)            
            que_pos = que_pos.to(device)            
            ans = ans.to(device)            
            ans_pos = ans_pos.to(device)            
            
            gnd = ans[:,1:].contiguous().view(-1).long()
            optimizer.zero_grad()
            pred = model(que.long(), que_pos, ans.long(), ans_pos)
            loss = loss_func(pred, gnd)
            epoch_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            iteration += 1
        scheduler.step()
        avg_epoch_loss = sum(epoch_loss) / len(epoch_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch: {epoch+1} iteration: {iteration} loss: {avg_epoch_loss} lr: {current_lr}")
    
    return model
        
            
def val(model, data_loader, loss_func, device="cpu"):
    n_correct = 0
    total = 0
    model.eval()
    for item in data_loader:
        que, que_pos, ans, ans_pos = item
        model.to(device)
        que = que.to(device) 
        que_pos = que_pos.to(device) 
        ans = ans.to(device) 
        ans_pos = ans_pos.to(device)
        gnd = ans[:,1:].contiguous().view(-1).long()
        
        pred = model(que.long(), que_pos, ans.long(), ans_pos)
        
        pred_max = pred.max(dim=1)[1]
        # print(pred.max(dim=1)[1].shape)
        # print(gnd.shape)
        # print(pred_max.eq(gnd).sum().item())
        n_correct += pred_max.eq(gnd).sum().item()
        total += len(pred_max)
    return n_correct / total

def inference(model : Transformers, que_exp, que_exp_pos, ans_exp_pos, ans_seq_len, device="cpu"):
    model.eval()
    exp_out = torch.LongTensor([14]).unsqueeze(0).to(device)
    
    que_emb = model.emb_layer(que_exp)
    que = que_emb + que_exp_pos
    enc_out = model.encoder(que)

    for _ in range(ans_seq_len - 1):
        ans_emb = model.emb_layer(exp_out)
        ans = ans_emb + ans_exp_pos[:, : exp_out.shape[1], :]
        pred = model.decoder(ans, enc_out, None)
        _, next_word = torch.max(pred[0, exp_out.shape[1]-1:exp_out.shape[1]] ,dim=1)
        
        exp_out = torch.cat([exp_out, next_word.view(1,1)], dim=1)
    
    return exp_out, model