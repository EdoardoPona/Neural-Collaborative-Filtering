import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim


class NeuralCollaborativeFiltering(torch.nn.Module):

    def __init__(self, user_num, item_num, mlp_embedding_size, gmf_embedding_size, mlp_hidden_size):
        super(NeuralCollaborativeFiltering, self).__init__()
        self.mlp_user_embeddings = nn.Embedding(num_embeddings=user_num, embedding_dim=mlp_embedding_size)
        self.mlp_item_embeddings = nn.Embedding(num_embeddings=item_num, embedding_dim=mlp_embedding_size)
        self.gmf_user_embeddings = nn.Embedding(num_embeddings=user_num, embedding_dim=gmf_embedding_size)
        self.gmf_item_embeddings = nn.Embedding(num_embeddings=item_num, embedding_dim=gmf_embedding_size)

        self.mlp = nn.Sequential([nn.Linear(mlp_embedding_size, mlp_hidden_size), nn.ReLU(),
                                  nn.Linear(mlp_hidden_size, mlp_hidden_size), nn.ReLU(),
                                  nn.Linear(mlp_hidden_size, mlp_hidden_size), nn.Sigmoid()])

        self.output_logits = nn.Linear(mlp_hidden_size + gmf_embedding_size, 1)

    def forward(self, user_id, item_id):
        user_emb = self.mlp_user_embeddings(user_id)
        item_emb = self.mlp_item_embeddings(item_id)

        mlp_output = self.mlp(torch.cat([user_emb, item_emb], dim=1))
        mf_outpt = torch.mul(self.gmf_user_embeddingsf(user_id), self.gmf_item_embeddings(item_id))

        out = torch.sigmoid(self.output_logits(torch.cat([mlp_output, mf_outpt])))
        return out
