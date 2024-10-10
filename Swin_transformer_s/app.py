from simple_swin_transformer import SimpleSwinTransformer
from patch_merge import SimplePatchMerging
from data import preprocess_cifar10
from train import trainer
from torch.utils.data import DataLoader

if __name__ == "__main__":
    # trainer params
    num_epochs = 5
    learning_rate = 1e-5
    device = "cpu"
    batch_size = 32
    validation_ratio = 0.2
    
    data_dict = preprocess_cifar10(1000, 100, validation_ratio)
    
    X_train = data_dict["X_train"]
    y_train = data_dict["y_train"]
    X_val = data_dict["X_val"]
    y_val = data_dict["y_val"] 
    
    X_test = data_dict["X_test"] 
    y_test = data_dict["y_test"] 

    X_train_set = DataLoader(X_train, batch_size=batch_size, shuffle=False, drop_last=True)
    y_train_set = DataLoader(y_train, batch_size=batch_size, shuffle=False, drop_last=True)
    X_val_set = DataLoader(X_val, batch_size=batch_size, shuffle=False, drop_last=True)
    y_val_set = DataLoader(y_val, batch_size=batch_size, shuffle=False, drop_last=True)
    X_test_set = DataLoader(X_test, batch_size=batch_size, shuffle=False, drop_last=True)
    y_test_set = DataLoader(y_test, batch_size=batch_size, shuffle=False, drop_last=True)

    print(len(X_train))
    swin_transformer = SimpleSwinTransformer(
        patch_size=[4,4],
        emb_dim=96,
        depths=[2,2,6,2],
        num_heads=[3,6,12,24],
        window_size=[4,4],
        stochastic_depth_prob=0.2,
        attention_dropout=0.1,
        dropout=0.1,
        mlp_ratio=4,
        downsample_layer=SimplePatchMerging,
        num_classes=10
    )
    
    trainer(
        model=swin_transformer,
        X_train_set=X_train_set,
        y_train_set=y_train_set,
        X_val_set=X_val_set,
        y_val_set=y_val_set,
        epochs=num_epochs,
        lr=learning_rate,
        early_stop=5,
        device=device
    )
    
    