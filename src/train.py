import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from src.model import VAE
from src.dataset import KITTIBEVDataset

def elbo_loss(recon, x, mu, logvar, beta=1.0):
    # BCE, much better for sparse BEV maps
    recon_loss = F.binary_cross_entropy(recon, x, reduction='sum') / x.size(0)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon_loss + beta * kl_loss, recon_loss, kl_loss


def train(epochs=100, latent_dim=256, batch_size=8, lr=1e-4, beta_max=0.1, resume_from = None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")

    # Dataset + split
    dataset = KITTIBEVDataset('data/kitti/data_object_velodyne/training/velodyne')
    val_size = int(0.1 * len(dataset))
    train_ds, val_ds = random_split(dataset, [len(dataset) - val_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4)

    model     = VAE(latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    start_epoch = 1
    if resume_from is not None:
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint)
        # Extract epoch number from filename e.g. 'vae_bce_epoch10.pt' → 10
        start_epoch = int(resume_from.split('epoch')[-1].replace('.pt', '')) + 1
        print(f"Resumed from {resume_from}, starting at epoch {start_epoch}")

    for epoch in range(start_epoch, epochs + 1):
        # KL annealing: ramp beta from 0 → beta_max over first 30 epochs
        # This prevents posterior collapse early in training
        beta = min(beta_max, beta_max * epoch / 30)

        # ── Train ──
        model.train()
        train_loss = 0
        for batch in train_loader:
            x = batch.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(x)
            loss, recon_l, kl_l = elbo_loss(recon, x, mu, logvar, beta)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # ── Validate ──
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch.to(device)
                recon, mu, logvar = model(x)
                loss, _, _ = elbo_loss(recon, x, mu, logvar, beta)
                val_loss += loss.item()

        print(f"Epoch {epoch:03d} | β={beta:.2f} | "
              f"Train: {train_loss/len(train_loader):.2f} | "
              f"Val: {val_loss/len(val_loader):.2f} | "
              f"Recon: {recon_l:.2f} | KL: {kl_l:.2f}")

        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            torch.save(model.state_dict(), f'outputs/bce_loss/vae_bce_epoch{epoch}.pt')

if __name__ == '__main__':
    train()