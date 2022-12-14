from triplet_selection import euclidean_distance_matrix,get_triplet_mask

class TripletLoss(nn.Module):
    """Uses all semi-hard triplets to compute Triplet loss
      Args:
        margin: Margin value in the Triplet Loss equation
    """
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        
    
    def forward(self, embeddings, labels):
        """computes loss value.
        Args:
          embeddings: Batch of embeddings, e.g., output of the encoder. shape: (batch_size, embedding_dim)
          labels: Batch of integer labels associated with embeddings. shape: (batch_size,)
        Returns:
          Scalar loss value.
        """
        distance_matrix = euclidean_distance_matrix(embeddings)

        distance_positive =  distance_matrix.unsqueeze(2)
        distance_negative = distance_matrix.unsqueeze(1)
        triplet_loss = distance_positive - distance_negative + self.margin
        triplet_loss=torch.where(distance_positive<distance_negative,triplet_loss,0)
        mask = get_triplet_mask(labels)
        triplet_loss *= mask
        triplet_loss = F.relu(triplet_loss)

     # step 4 - compute scalar loss value by averaging positive losses
        num_positive_losses = (triplet_loss > eps).float().sum()
        triplet_loss = triplet_loss.sum() / (num_positive_losses + eps)
        

        return triplet_loss
