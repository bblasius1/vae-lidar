import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=256):
        super().__init__()
        # Progressively downsample: 512 → 256 → 128 → 64 → 32 → 16
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32,  4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32,          64,  4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64,          128, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128,         256, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(256,         512, 4, stride=2, padding=1), nn.ReLU(),
        )
        self.flatten_dim = 512 * 16 * 16  # for 512×512 input
        self.fc_mu     = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

    def forward(self, x):
        h = self.conv(x).view(x.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    def __init__(self, out_channels=3, latent_dim=256):
        super().__init__()
        self.flatten_dim = 512 * 16 * 16
        self.fc = nn.Linear(latent_dim, self.flatten_dim)
        # Progressively upsample: 16 → 32 → 64 → 128 → 256 → 512
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64,  4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64,  32,  4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32,  out_channels, 4, stride=2, padding=1),
            nn.Sigmoid()  # output in [0, 1] to match normalized BEV
        )

    def forward(self, z):
        h = self.fc(z).view(z.size(0), 512, 16, 16)
        return self.deconv(h)


class VAE(nn.Module):
    def __init__(self, in_channels=3, latent_dim=256):
        super().__init__()
        self.encoder = Encoder(in_channels, latent_dim)
        self.decoder = Decoder(in_channels, latent_dim)

    def reparameterize(self, mu, logvar):
        # Reparameterization trick: z = mu + eps * std
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

    def sample(self, n, device):
        """Generate n new BEV scenes from random latent vectors."""
        z = torch.randn(n, self.encoder.fc_mu.out_features).to(device)
        with torch.no_grad():
            return self.decoder(z)