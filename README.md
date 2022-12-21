
# Facenet and Siamese Implementation

In this using paper Facenet: A Unified Embedding for Face Recognition and Clustering we implemented it on pytorch using dataset Labeled Faces in the Wild.

The second part involves the implementation of standard siamese network on same dataset Labeled faces in the wild .



## File Description
**helpers.py**: it contains following functions:

   **pairwise_distances**:Compute the 2D matrix of distances between all the embeddings.

   **get_triplet_mask(labels)**:Return a 3D mask where mask[a,p,n] is True iff the triplet (a, p, n) is valid.

   **get_anchor_positive_triplet_mask()**:Return a 2D mask where mask[a, p] is True iff a and p are distinct and have same label.

   **get_anchor_negative_triplet_mask()**:Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.

   **Class BalancedBatchSampler()**:Returns batches of size n_classes * n_samples

**layers.py**: It contains following two classes:

   **Inception_block()**:Inception_block Class is used to create an Inception Module object similar to the Inception module described in the Fig 2(b) of "Going Deeper     with Convolutions" paper.
        
    The two major differences are:
        1. There is an option to choose which layers to apply from the four layers generally used.
        2. Pooling can be chosen to be "Max Pool" or "L2 Pool", and also dimensionality reduction can be applied if needed

  **conv_block()**:conv_block Class is used to create a 2-D Convolution layer with batch normalization object.After applying 2-D Convolution, batch normalization and        ReLU activation has been applied.

**loss.py**:
batch_hard_triplet_loss():Build the triplet loss over a batch of embeddings.For each anchor, we get the hardest positive and hardest negative to form a triplet.

**models.py**: it contains following models:

   **nn2**: nn2 Class is used to create an Inception network object with the architecture of NN2 (inception_224x224) given in "FaceNet". It takes as input a RGB image    of size 224x224x3.

   **nn3**: nn2 Class is used to create an Inception network object with the architecture of NN2 (inception_160x160) given in "FaceNet". It takes as input a RGB image    of size 160x160x3.

   **nn4**: nn2 Class is used to create an Inception network object with the architecture of NN2 (inception_96x96) given in "FaceNet". It takes as input a RGB image of    size 96x96x3.

   **nns**: nn2 Class is used to create an Inception network object with the architecture of NN2 (inception_165x165) given in "FaceNet". It takes as input a RGB image     of size 165x165x3.

**train.py**: training different models and saving the models after being trained on celeba dataset.

**train_helpers.py**: it contains functions that will help in training







## Documentation

[Pytorch Documentation](https://pytorch.org/docs/stable/index.html)


## Optimizations

We used Adam algorithm to optimize the Siamese Network model and used the Adagrad algorithm to optimize the Facenet models.


## Acknowledgements

 - [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/abs/1503.03832)
 - [Siamese Neural Networks for One-shot Image Recognition](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)
 - [Triplet Loss — Advanced Intro](https://towardsdatascience.com/triplet-loss-advanced-intro-49a07b7d8905#:%7E:text=This%20is%20due%20to%20the,pairs%20are%20at%20that%20moment)


## Authors

- [Anadi](https://github.com/Anadigoyal)
- [Akhil](https://github.com/Akhil-blanc)
- [Shubh](https://github.com/Shubh-Goyal-07)
- [Sukriti](https://github.com/s-goyal23)
- [Rushi](https://github.com/shahrushi2003)
- [Ritik](https://github.com/testgithubtiwari)
- [Ameen](https://github.com/AmeenRizvi)







