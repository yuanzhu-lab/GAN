# %%
from __future__ import print_function
# Set-up
import os

path = r"./"
if os.path.exists(path):
    print("ok")

# %%
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
from IPython.display import Image, display
import matplotlib.pyplot as plt
import os

if not os.path.exists('results'):
    os.mkdir('results')

batch_size = 100
latent_size = 20

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True, **kwargs)


class Generator(nn.Module):
    # The generator takes an input of size latent_size, and will produce an output of size 784.
    # It should have a single hidden linear layer with 400 nodes using ReLU activations, and use Sigmoid activation for its outputs
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(latent_size, 400)
        self.fc2 = nn.Linear(400, 784)

    def forward(self, z):
        h = torch.relu(self.fc1(z))
        x_fake = torch.sigmoid(self.fc2(h))
        return x_fake


class Discriminator(nn.Module):
    # The discriminator takes an input of size 784, and will produce an output of size 1.
    # It should have a single hidden linear layer with 400 nodes using ReLU activations, and use Sigmoid activation for its output
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc2 = nn.Linear(400, 1)

    def forward(self, x):
        # x can be (batch, 1, 28, 28) or (batch, 784)
        x = x.view(-1, 784)
        h = torch.relu(self.fc1(x))
        out = torch.sigmoid(self.fc2(h))
        return out


bce_loss = nn.BCELoss()


def train(generator, generator_optimizer, discriminator, discriminator_optimizer):
    # Trains both the generator and discriminator for one epoch on the training dataset.
    # Returns the average generator and discriminator loss (scalar values, use the binary cross-entropy appropriately)
    generator.train()
    discriminator.train()

    total_gen_loss = 0.0
    total_disc_loss = 0.0

    for _, (real_imgs, _) in enumerate(train_loader):
        real_imgs = real_imgs.to(device)
        batch_size_curr = real_imgs.size(0)

        real_labels = torch.ones(batch_size_curr, 1, device=device)
        fake_labels = torch.zeros(batch_size_curr, 1, device=device)

        discriminator_optimizer.zero_grad()

        real_outputs = discriminator(real_imgs)
        d_loss_real = bce_loss(real_outputs, real_labels)

        z = torch.randn(batch_size_curr, latent_size, device=device)
        fake_imgs = generator(z)
        fake_outputs = discriminator(fake_imgs.detach())
        d_loss_fake = bce_loss(fake_outputs, fake_labels)

        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        discriminator_optimizer.step()

        generator_optimizer.zero_grad()

        z = torch.randn(batch_size_curr, latent_size, device=device)
        fake_imgs = generator(z)
        outputs = discriminator(fake_imgs)
        g_loss = bce_loss(outputs, real_labels)

        g_loss.backward()
        generator_optimizer.step()

        total_gen_loss += g_loss.item()
        total_disc_loss += d_loss.item()

    avg_generator_loss = total_gen_loss / len(train_loader)
    avg_discriminator_loss = total_disc_loss / len(train_loader)
    return avg_generator_loss, avg_discriminator_loss


def test(generator, discriminator):
    # Runs both the generator and discriminator over the test dataset.
    # Returns the average generator and discriminator loss (scalar values, use the binary cross-entropy appropriately)
    generator.eval()
    discriminator.eval()

    total_gen_loss = 0.0
    total_disc_loss = 0.0

    with torch.no_grad():
        for real_imgs, _ in test_loader:
            real_imgs = real_imgs.to(device)
            batch_size_curr = real_imgs.size(0)

            real_labels = torch.ones(batch_size_curr, 1, device=device)
            fake_labels = torch.zeros(batch_size_curr, 1, device=device)

            real_outputs = discriminator(real_imgs)
            d_loss_real = bce_loss(real_outputs, real_labels)

            z = torch.randn(batch_size_curr, latent_size, device=device)
            fake_imgs = generator(z)
            fake_outputs = discriminator(fake_imgs)

            d_loss_fake = bce_loss(fake_outputs, fake_labels)

            d_loss = d_loss_real + d_loss_fake

            # Generator loss (try to fool discriminator)
            g_loss = bce_loss(fake_outputs, real_labels)

            total_gen_loss += g_loss.item()
            total_disc_loss += d_loss.item()

    avg_generator_loss = total_gen_loss / len(test_loader)
    avg_discriminator_loss = total_disc_loss / len(test_loader)
    return avg_generator_loss, avg_discriminator_loss


epochs = 50

discriminator_avg_train_losses = []
discriminator_avg_test_losses = []
generator_avg_train_losses = []
generator_avg_test_losses = []

generator = Generator().to(device)
discriminator = Discriminator().to(device)

generator_optimizer = optim.Adam(generator.parameters(), lr=1e-3)
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=1e-3)

for epoch in range(1, epochs + 1):
    generator_avg_train_loss, discriminator_avg_train_loss = train(
        generator, generator_optimizer, discriminator, discriminator_optimizer
    )
    generator_avg_test_loss, discriminator_avg_test_loss = test(generator, discriminator)

    discriminator_avg_train_losses.append(discriminator_avg_train_loss)
    generator_avg_train_losses.append(generator_avg_train_loss)
    discriminator_avg_test_losses.append(discriminator_avg_test_loss)
    generator_avg_test_losses.append(generator_avg_test_loss)

    with torch.no_grad():
        sample = torch.randn(64, latent_size).to(device)
        sample = generator(sample).cpu()
        save_image(sample.view(64, 1, 28, 28),
                   'results/sample_' + str(epoch) + '.png')
        print('Epoch #' + str(epoch))
        display(Image('results/sample_' + str(epoch) + '.png'))
        print('\n')

plt.plot(discriminator_avg_train_losses)
plt.plot(generator_avg_train_losses)
plt.title('Training Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Disc', 'Gen'], loc='upper right')
plt.show()

plt.plot(discriminator_avg_test_losses)
plt.plot(generator_avg_test_losses)
plt.title('Test Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Disc', 'Gen'], loc='upper right')
plt.show()
