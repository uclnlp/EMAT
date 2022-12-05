import torch
from torch import nn


class FusionWeight(nn.Module):
    # ["cat_k+v_g(kq)", "cat_k+v_g(kv)"]:
    def __init__(self, fusion_type="cat_k+v_g(kq)", model_dim=512, prefix_length=2, key_dim=1024):
        super(FusionWeight, self).__init__()
        self.fusion_type = fusion_type
        if "g(kq)" in self.fusion_type:
            self.score_proj = nn.Linear(1, 1)
        else:
            input_dim = model_dim * prefix_length + key_dim
            self.score_proj = nn.Linear(input_dim, 1)

    def forward(self, key=None, query=None, value=None):
        if "g(kv)" in self.fusion_type:
            # key: batch_size, num_values, key_token_num, hidden_size
            # value: batch_size, num_values, prefix_length, hidden_size
            batch_size, num_values, _, _ = key.shape
            input_hidden = torch.cat((key, value), dim=2)  # batch_size, num_values, key_token_num+prefix_length, hidden
            scores = self.score_proj(input_hidden.view(batch_size, num_values, -1))
        else:
            # key: batch_size, num_values, hidden_size
            # query: batch_size, hidden_size
            scores = torch.bmm(key, query.unsqueeze(dim=1).transpose(2, 1))  # batch_size, num_values, 1
            scores = self.score_proj(scores)

        scores = torch.sigmoid(scores)
        return scores
