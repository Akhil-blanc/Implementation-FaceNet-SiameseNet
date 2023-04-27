import torch
import torch.nn as nn
import torch.nn.functional as F

eps = 1e-8 

def euclidean_distance_matrix(x):
  """Efficient computation of Euclidean distance matrix
  Args:
    x: Input tensor of shape (batch_size, embedding_dim)
    
  Returns:
    Distance matrix of shape (batch_size, batch_size)
  """

  dot_product = torch.mm(x, x.t())


  squared_norm = torch.diag(dot_product)


  distance_matrix = squared_norm.unsqueeze(0) - 2 * dot_product + squared_norm.unsqueeze(1)


  distance_matrix = F.relu(distance_matrix)


  mask = torch.where(distance_matrix==0.0, 1., 0.)


  distance_matrix += mask * eps


  distance_matrix = torch.sqrt(distance_matrix)


  distance_matrix *= (1.0 - mask)

  return distance_matrix

def get_triplet_mask(labels):
  """compute a mask for valid triplets
  Args:
    labels: Batch of integer labels. shape: (batch_size,)
  Returns:
    Mask tensor to indicate which triplets are actually valid. Shape: (batch_size, batch_size, batch_size)
    A triplet is valid if:
    `labels[i] == labels[j] and labels[i] != labels[k]`
    and `i`, `j`, `k` are different.
  """
  
  indices_equal = torch.eye(labels.size()[0], dtype=torch.bool, device=labels.device)
  indices_not_equal = torch.logical_not(indices_equal)
  
  i_not_equal_j = indices_not_equal.unsqueeze(2)
  
  i_not_equal_k = indices_not_equal.unsqueeze(1)
  
  j_not_equal_k = indices_not_equal.unsqueeze(0)
  
  distinct_indices = torch.logical_and(torch.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)

  
  labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
  
  i_equal_j = labels_equal.unsqueeze(2)
  
  i_equal_k = labels_equal.unsqueeze(1)
  
  valid_indices = torch.logical_and(i_equal_j, torch.logical_not(i_equal_k))

  
  mask = torch.logical_and(distinct_indices, valid_indices)

  return mask

