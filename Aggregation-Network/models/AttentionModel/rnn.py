import torch.nn as nn


class SimpleRNN(nn.Module):

    def __init__(self, in_dim=128, hidden_dim=96, n_layers=2, drop_out=0.5, num_classes=1000):
        super(SimpleRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.fea_conv = nn.Sequential(nn.Dropout2d(drop_out),
                                      nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False),
                                      nn.BatchNorm2d(256),
                                      nn.ReLU(),
                                      nn.Dropout2d(drop_out),
                                      nn.Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False),
                                      nn.BatchNorm2d(128),
                                      nn.ReLU(),
                                      nn.Dropout2d(drop_out),
                                      )
        self.fea_lstm = nn.GRU(in_dim, hidden_dim, num_layers=self.n_layers, batch_first=True, bidirectional=True)
        self.fea_first_final = nn.Sequential(nn.Conv2d(128, hidden_dim * 2, kernel_size=(24, 1), stride=(1, 1), padding=(0, 0), bias=True))
        self.fea_secon_final = nn.Sequential(nn.Conv2d(hidden_dim * 2, hidden_dim * 2, kernel_size=(24, 1), stride=(1, 1), padding=(0, 0), bias=True))
        self.classifier = nn.Linear(hidden_dim * 2, num_classes, bias=False)

    def forward(self, data):
        x, pro = data
        bs, seq_len, fea_len = x.shape
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(bs, fea_len, seq_len, 1).contiguous()
        x_out = self.fea_conv(x)
        x0 = self.fea_first_final(x_out)
        x0 = x0.view(bs, -1).contiguous()

        x_out = x_out.view(bs, 128, seq_len).contiguous()
        x1 = x_out.permute(0, 2, 1).contiguous()
        x1, _ = self.fea_lstm(x1)
        x1 = x1.view(bs, 1, seq_len, self.hidden_dim * 2).contiguous()
        x1 = x1.permute(0, 3, 2, 1).contiguous()
        x1 = self.fea_secon_final(x1)
        x1 = x1.view(bs, -1).contiguous()
        out0 = x0 + x1

        out = self.classifier(out0)

        return out
