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

        # TODO the paper uses a tower structure, halving the layer size each time
        self.mlp = nn.Sequential(nn.Linear(2*mlp_embedding_size, mlp_hidden_size), nn.ReLU(),
                                  nn.Linear(mlp_hidden_size, mlp_hidden_size), nn.ReLU(),
                                  nn.Linear(mlp_hidden_size, mlp_hidden_size), nn.ReLU())

        self.gmf_out = nn.Linear(gmf_embedding_size, 1, bias=False)
        self.mlp_out = nn.Linear(mlp_hidden_size, 1, bias=False)

        self.output_logits = nn.Linear(mlp_hidden_size + gmf_embedding_size, 1, bias=False)
        self.model_blending = 0.5           # alpha parameter, equation 13 in the paper

    def forward(self, x, mode='ncf'):
        user_id, item_id = x[:, 0], x[:, 1]
        if mode == 'ncf':
            gmf_product = self.gmf_forward(user_id, item_id)
            mlp_output = self.mlp_forward(user_id, item_id)
            return torch.sigmoid(self.output_logits(torch.cat([gmf_product, mlp_output], dim=1)))
        elif mode == 'mlp':
            return torch.sigmoid(self.mlp_out(self.mlp_forward(user_id, item_id)))
        elif mode == 'gmf':
            return torch.sigmoid(self.gmf_out(self.gmf_forward(user_id, item_id)))

    def gmf_forward(self, user_id, item_id):
        user_emb = self.gmf_user_embeddings(user_id)
        item_emb = self.gmf_item_embeddings(item_id)
        return torch.mul(user_emb, item_emb)

    def mlp_forward(self, user_id, item_id):
        user_emb = self.mlp_user_embeddings(user_id)
        item_emb = self.mlp_item_embeddings(item_id)
        return self.mlp(torch.cat([user_emb, item_emb], dim=1))

    def join_output_weights(self):
        """ join the last layer after pretraining """
        W = nn.Parameter(torch.cat((self.model_blending*self.gmf_out.weight,
                                               (1-self.model_blending)*self.mlp_out.weight), dim=1))

        self.output_logits.weight = W

