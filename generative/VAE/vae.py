import torch
import  torch.nn.functional as F
from torch import nn
from torchvision.ops import Permute
from data import load_cifar

def reparametrize(mu, logvar):
    std = torch.exp(logvar * 0.5)
    epsilon = torch.randn_like(std)
    z = std * epsilon + mu
    return z

def loss_function(x_hat, x, mu, logvar):
    N = x.size(0)
    reconstruction =  F.binary_cross_entropy(x_hat, x, reduction="sum")
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    loss = reconstruction + kl_divergence
    loss /= N
    return loss

class ConvLayer(nn.Module):
    
    def __init__(
        self,
        dim
    ):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim),
            Permute([0,2,3,1]),
            nn.LayerNorm(dim),
            nn.Linear(dim, dim*4),
            nn.GELU(),
            nn.Linear(dim*4, dim),
            Permute([0,3,1,2])
        )
        layer_scale = 1e-6
        self.layer_scale = nn.Parameter(torch.ones(dim, 1, 1) * layer_scale)
    
    def forward(self, x):
        return x + (self.layers(x) * self.layer_scale)

class LayerNorm2d(nn.LayerNorm):
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x
    
class VariationalAutoEncoder(nn.Module):
    
    def __init__(
        self, 
        z_size,
        hidden_size
    ):
        super().__init__()
        
        # in, out, n_layers
        conv_settings = [
            (3, 96, 2),
            (96, 192, 2),
            (192, 384, 4),
            (384, None, 2),
        ]
        
        self.encoder = None
        self.mu_layer = None
        self.logvar = None
        self.decoder = None
        
        
        layers = []
        for setting in conv_settings:
            in_channel, out_channel, n_layers = setting
            for _ in range(n_layers):
                layers.append(
                    ConvLayer(
                        in_channel
                    )
                )
            if out_channel is not None:
                layers.append(
                    nn.Sequential(
                        LayerNorm2d(in_channel),
                        nn.Conv2d(in_channel, out_channel, kernel_size=2, stride=2),
                    )
                )
                
        layers.append(nn.Sequential(
            nn.Flatten(),
            nn.Linear(4*4*384, hidden_size),
            nn.ReLU()
        ))
        
        self.encoder = nn.Sequential(*layers)
        self.mu_layer = nn.Linear(hidden_size, z_size)
        self.logvar = nn.Linear(hidden_size, z_size)
        
        decode_layers = []
        
        decode_layers.append(
            nn.Sequential(
                nn.Linear(z_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 4*4*384),
                nn.ReLU(),
            )
        )

        decode_layers.append(nn.Unflatten(1, (384, 4, 4)))
        
        for setting in reversed(conv_settings):
            in_channel, out_channel, n_layers = setting
            
            if out_channel is not None:
                decode_layers.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(out_channel, in_channel, kernel_size=2, stride=2),
                        LayerNorm2d(in_channel),
                    )
                )
            #  possibly add ConvLayer
        
        self.decoder = nn.Sequential(*decode_layers)
        
    def forward(self, x):
        enc_out = self.encoder(x)
        mu = self.mu_layer(enc_out)
        logvar = self.logvar(enc_out)
        z = reparametrize(mu, logvar)
        x_hat = self.decoder(z)
        x_hat = F.sigmoid(x_hat)
        return x_hat, mu, logvar
    
def train_vae(epoch, model, train_loader, cond=False, device="cpu"):
    model.train()
    train_loss = 0
    num_classes = 10
    loss = None
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for batch_idx, (data, labels) in enumerate(train_loader):
        data = data.to(device=device)
        if cond:
            one_hot_vec = torch.one_hot(labels, num_classes).to(device=device)
            recon_batch, mu, logvar = model(data, one_hot_vec)
        else:
            recon_batch, mu, logvar = model(data)
        optimizer.zero_grad()
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.data
        optimizer.step()
    print("Train Epoch: {} \tLoss: {:.6f}".format(epoch, loss.data))
    
if __name__ == "__main__":
    latent_size = 20
    model = VariationalAutoEncoder(latent_size, 400)
    train_loader, val_loader, test_loader = load_cifar()
    num_epochs = 10
    for epoch in range(0, num_epochs):
        train_vae(1, model, train_loader)
        
    z = torch.randn(10, latent_size).to(device='cuda')
    import matplotlib.gridspec as gridspec
    import matplotlib as plt
    model.eval()
    print(z.shape)
    samples = model.decoder(z).data.cpu().numpy()

    fig = plt.figure(figsize=(10, 1))
    gspec = gridspec.GridSpec(1, 10)
    gspec.update(wspace=0.05, hspace=0.05)
    for i, sample in enumerate(samples):
        ax = plt.subplot(gspec[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28,28), cmap='Greys_r')
