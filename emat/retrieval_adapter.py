from torch import nn


class RetAdapter(nn.Module):

    def __init__(self, in_dim, out_dim, adapter_type="linear"):
        super(RetAdapter, self).__init__()
        assert adapter_type in ["linear", "dropout_linear"]
        if adapter_type == "linear":
            self.out = nn.Linear(in_dim, out_dim, bias=True)
        elif adapter_type == "dropout_linear":
            self.out = nn.Sequential(
                nn.Dropout(0.9),
                nn.Linear(in_dim, out_dim, bias=True)
            )
        else:
            raise NotImplementedError

    def forward(self, vectors):
        return self.out(vectors)
