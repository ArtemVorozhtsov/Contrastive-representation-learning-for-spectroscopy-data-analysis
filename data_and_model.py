from torch.utils.data import Dataset

rng = np.random.default_rng()

class SiameseDataset(Dataset):

  def __init__(self, x, y, transform = None, number_of_samples = 20):
    self.x = x
    self.transform = transform
    self.y = y.reshape(-1)
    self.n = number_of_samples
    self.classes = np.arange(500)

  def __getitem__(self, idx):
    anchor = self.x[idx//self.n][idx % self.n]
    positive = rng.choice(self.x[idx//self.n])
    negative_idxs = np.delete(self.classes, self.classes == idx//self.n)
    negative_idx = np.random.choice(negative_idxs)
    negative = rng.choice(self.x[negative_idx])

    if self.transform is not None:
      anchor = self.transform(anchor)
      positive = self.transform(positive)
      negative = self.transform(negative)
    return anchor, positive, negative


  def __len__(self):
    return len(self.x)*self.n

from torch.utils.data import Dataset

class ValidationSiameseDataset(Dataset):

  def __init__(self, x, y, transform = None, number_of_samples = 10):
    self.x = x
    self.transform = transform
    self.y = y.reshape(-1)
    self.n = number_of_samples
    self.classes = np.arange(500)

  def __getitem__(self, idx):
    anchor = self.x[idx//self.n][idx % self.n]
    positive = rng.choice(self.x[idx//self.n])
    negative_idxs = np.delete(self.classes, self.classes == idx//self.n)
    negative_idx = np.random.choice(negative_idxs)
    negative = rng.choice(self.x[negative_idx])

    if self.transform is not None:
      anchor = self.transform(anchor)
      positive = self.transform(positive)
      negative = self.transform(negative)
    return anchor, positive, negative


  def __len__(self):
    return len(self.x)*self.n

def transformation(x):
  x = np.abs(x)
  x = x/np.max(x)
  x = x + np.random.normal(0,0.005,x.shape[0])
  x = torch.Tensor(x)
  return x

from torch import nn
class SiameseNet(nn.Module):

    def __init__(self, latent_dim):
      super().__init__()
      self.latent_dim = latent_dim
      self.model = nn.Sequential(nn.Conv1d(in_channels=1, out_channels=16,kernel_size=21, padding = 'same'),
                                  nn.MaxPool1d(kernel_size=2, stride=2),
                                  nn.LeakyReLU(),
                                  nn.BatchNorm1d(16),
                                  nn.Conv1d(16, 32, 11, padding = 'same'),
                                  nn.MaxPool1d(2, 2),
                                  nn.LeakyReLU(),
                                  nn.BatchNorm1d(32),
                                  nn.Conv1d(32, 64, 5, padding = 'same'),
                                  nn.MaxPool1d(2, 2),
                                  nn.LeakyReLU(),
                                  nn.BatchNorm1d(64),

                                 nn.Flatten(),
                                 nn.Linear(64*625, self.latent_dim))


    def _forward(self, x):
      out = x.view(-1, 1, 5000)
      out = self.model(out)
      # normalize embedding to unit vector
      out = torch.nn.functional.normalize(out)
      return out


    def predict(self, x):
      out = x.view(-1, 1, 5000)
      out = self.model(out)
      out = torch.nn.functional.normalize(out)
      return out

    def forward(self, anchor, positive, negative, latent_dim):
        output1 = self._forward(anchor)
        output2 = self._forward(positive)
        output3 = self._forward(negative)

        return output1, output2, output3


def train(num_epochs, model, criterion_train, criterion_val, optimizer, train_loader, val_loader, latent_dim):
    loss_history = []
    l = []
    l_val = []
    f_p_train = []
    f_p_val = []

    for epoch in range(0, num_epochs):
      model.train()
      for i, batch in enumerate(train_loader, 0):
          anc, pos, neg = batch
          output_anc, output_pos, output_neg = model(anc.to(device), pos.to(device), neg.to(device), latent_dim)
          loss, fraction_pos = criterion_train(output_anc, output_pos, output_neg)
          loss.backward()
          optimizer.step()
          optimizer.zero_grad()
          l.append(loss.item())
          f_p_train.append(fraction_pos.item())

      model.eval()
      for i, batch in enumerate(val_loader, 0):
          anc, pos, neg = batch
          output_anc, output_pos, output_neg = model(anc.to(device), pos.to(device), neg.to(device), latent_dim)
          loss_val, fraction_pos_val = criterion_val(output_anc, output_pos, output_neg)
          l_val.append(loss_val.item())
          f_p_val.append(fraction_pos_val.item())
      scheduler.step()
      f_p_train_last_ep =  torch.tensor(f_p_train[-len(train_loader):-1]).mean()
      f_p_val_last_ep = torch.tensor(f_p_val[-len(val_loader):-1]).mean()
      print("Epoch {} with {:.4f} fraction and {:.4f} val_fraction".format(epoch, f_p_train_last_ep, f_p_val_last_ep))

    return l, l_val, f_p_train, f_p_val

import torch.nn.functional as F
class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin, semi_hard):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.semi_hard = semi_hard

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = 1.0 - F.cosine_similarity(anchor, positive)
        distance_negative = 1.0 - F.cosine_similarity(anchor, negative)
        losses = distance_positive - distance_negative + self.margin
        losses = torch.where(losses > self.semi_hard, losses, torch.zeros(losses.shape).to(device))
        return losses.sum()/(torch.count_nonzero(losses)+1e-16) if size_average else losses.sum(), torch.count_nonzero(losses)/len(losses)
