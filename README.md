In this work, we establish the Siamese network LLPS models which can integrates the ESM2 embedding features and the Protein-Protein interaction networks embedding features for LLPS prediction.

Detailed description:

deepNF_embedding.py : It can be used to created deep autoencoder models to extract the embedding features from Protein-Protein interaction networks.

node2vec_embedding.py: It can be used to create node2vec models to extract the embedding features from Protein-Protein interaction networks.

Siamese_network_LLPS.py: It can be used to create models based on Siamese networks for predicting liquid-liquid phase separation of proteins.

positive_protein_set.csv: It is the 876  liquid-liquid phase separation of proteins.

negative_protein_set.csv: It is the 1560  highly unlikely to undergo liquid-liquid phase separation of proteins.

ESM2_feature_set.zip: It is the esm2 embedding features of banchmark dataset.

deepNF-feature_set.zip: It is the embedding features extracted from the protein-protein interaction network of the benchmark dataset using the deep autoencoder models.

node2vec-feature_set.zip: It is the embedding features extracted from the protein-protein interaction network of the benchmark dataset using the node2vec method.
