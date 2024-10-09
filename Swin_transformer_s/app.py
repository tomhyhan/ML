from simple_swin_transformer import SimpleSwinTransformer
from patch_merge import SimplePatchMerging


if __name__ == "__main__":
    # trainer params
    num_epochs = 2
    learning_rate = 1e-5
    
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
    
    
    
    pass