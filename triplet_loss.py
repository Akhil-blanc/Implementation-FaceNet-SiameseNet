# -*- coding: utf-8 -*-
"""triplet_loss.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1e8DmkxucaC_lgBbKHFmrdSXboUDgiN9U
"""

from triplet_selection import euclidean_distance_matrix,get_triplet_mask
class TripletLoss(nn.Module):
  
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        
    def euclidean(self, x1, x2):
        return (x1 - x2).pow(2).sum(1)
    
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        distance_matrix = euclidean_distance_matrix(embeddings)

        distance_positive =  distance_matrix.unsqueeze(2)
        distance_negative = distance_matrix.unsqueeze(1)
        triplet_loss = distance_positive - distance_negative + self.margin
        triplet_loss=torch.where(distance_positive<distance_negative,losses,0)
        mask = get_triplet_mask(labels)
        triplet_loss *= mask
        triplet_loss = F.relu(triplet_loss)

     # step 4 - compute scalar loss value by averaging positive losses
        num_positive_losses = (triplet_loss > eps).float().sum()
        triplet_loss = triplet_loss.sum() / (num_positive_losses + eps)
        

        return triplet_loss