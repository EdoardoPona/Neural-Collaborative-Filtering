import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim


class NeuralCollaborativeFiltering(torch.nn.Module):

    def __init__(self, user_num, item_num, predictive_factor):
        super(NeuralCollaborativeFiltering, self).__init__()
        self.mlp_user_embeddings = nn.Embedding(num_embeddings=user_num, embedding_dim=2*predictive_factor)
        self.mlp_item_embeddings = nn.Embedding(num_embeddings=item_num, embedding_dim=2*predictive_factor)

        self.gmf_user_embeddings = nn.Embedding(num_embeddings=user_num, embedding_dim=2*predictive_factor)
        self.gmf_item_embeddings = nn.Embedding(num_embeddings=item_num, embedding_dim=2*predictive_factor)


        self.mlp = nn.Sequential(nn.Linear(4*predictive_factor, 2*predictive_factor), nn.ReLU(),
                                  nn.Linear(2*predictive_factor, predictive_factor), nn.ReLU(),
                                  nn.Linear(predictive_factor, predictive_factor//2), nn.ReLU())

        self.gmf_out = nn.Linear(2*predictive_factor, 1)
        self.gmf_out.weight = nn.Parameter(torch.ones(1, 2*predictive_factor))

        self.mlp_out = nn.Linear(predictive_factor//2, 1)

        self.output_logits = nn.Linear(predictive_factor, 1)
        self.model_blending = 0.5           # alpha parameter, equation 13 in the paper

        self.initialize_weights()

    def initialize_weights(self):

        nn.init.normal_(self.mlp_user_embeddings.weight, std=0.01)
        nn.init.normal_(self.mlp_item_embeddings.weight, std=0.01)
        nn.init.normal_(self.gmf_user_embeddings.weight, std=0.01)
        nn.init.normal_(self.gmf_item_embeddings.weight, std=0.01)

        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

        nn.init.kaiming_uniform_(self.gmf_out.weight, a=1, nonlinearity='sigmoid')
        nn.init.kaiming_uniform_(self.mlp_out.weight, a=1, nonlinearity='sigmoid')

    def forward(self, x, mode='ncf'):
        user_id, item_id = x[:, 0], x[:, 1]
        if mode == 'ncf':
            gmf_product = self.gmf_forward(user_id, item_id)
            mlp_output = self.mlp_forward(user_id, item_id)
            return torch.sigmoid(self.output_logits(torch.cat([gmf_product, mlp_output], dim=1)))
        elif mode == 'mlp':
            return torch.sigmoid(self.mlp_out(self.mlp_forward(user_id, item_id))).view(-1)
        elif mode == 'gmf':
            return torch.sigmoid(self.gmf_out(self.gmf_forward(user_id, item_id))).view(-1)

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



class NCF_item_item(torch.nn.Module):

    def __init__(self, item_num, predictive_factor):
        super(NCF_item_item, self).__init__()
        self.mlp_item_embeddings = nn.Embedding(num_embeddings=item_num, embedding_dim=2*predictive_factor)
        self.gmf_item_embeddings = nn.Embedding(num_embeddings=item_num, embedding_dim=2*predictive_factor)


        self.mlp = nn.Sequential(nn.Linear(4*predictive_factor, 2*predictive_factor), nn.ReLU(),
                                  nn.Linear(2*predictive_factor, predictive_factor), nn.ReLU(),
                                  nn.Linear(predictive_factor, predictive_factor//2), nn.ReLU())

        self.gmf_out = nn.Linear(2*predictive_factor, 1)
        self.gmf_out.weight = nn.Parameter(torch.ones(1, 2*predictive_factor))

        self.mlp_out = nn.Linear(predictive_factor//2, 1)

        self.output_logits = nn.Linear(predictive_factor, 1)
        self.model_blending = 0.5           # alpha parameter, equation 13 in the paper

        self.initialize_weights()

    def initialize_weights(self):

        nn.init.normal_(self.mlp_item_embeddings.weight, std=0.01)
        nn.init.normal_(self.gmf_item_embeddings.weight, std=0.01)

        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

        nn.init.kaiming_uniform_(self.gmf_out.weight, a=1, nonlinearity='sigmoid')
        nn.init.kaiming_uniform_(self.mlp_out.weight, a=1, nonlinearity='sigmoid')

    def forward(self, x, mode='ncf'):
        item0_id, item1_id = x[:, 0], x[:, 1]
        if mode == 'ncf':
            gmf_product = self.gmf_forward(item0_id, item1_id)
            mlp_output = self.mlp_forward(item0_id, item1_id)
            return torch.sigmoid(self.output_logits(torch.cat([gmf_product, mlp_output], dim=1)))
        elif mode == 'mlp':
            return torch.sigmoid(self.mlp_out(self.mlp_forward(item0_id, item1_id))).view(-1)
        elif mode == 'gmf':
            return torch.sigmoid(self.gmf_out(self.gmf_forward(item0_id, item1_id))).view(-1)

    def gmf_forward(self, item0_id, item1_id):
        # TODO       I am sure embeddings can be called once for n embeddings
        item0_emb = self.gmf_item_embeddings(item0_id)
        item1_emb = self.gmf_item_embeddings(item1_id)
        return torch.mul(item0_emb, item1_emb)

    def mlp_forward(self, item0_id, item1_id):
        # TODO       I am sure embeddings can be called once for n embeddings
        item0_emb = self.mlp_item_embeddings(item0_id)
        item1_emb = self.mlp_item_embeddings(item1_id)
        return self.mlp(torch.cat([item0_emb, item1_emb], dim=1))

    def join_output_weights(self):
        """ join the last layer after pretraining """
        W = nn.Parameter(torch.cat((self.model_blending*self.gmf_out.weight,
                                               (1-self.model_blending)*self.mlp_out.weight), dim=1))

        self.output_logits.weight = W
