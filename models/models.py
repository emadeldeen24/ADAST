import torch
from torch import nn


## Convolutional blocks
class base_model_blk1(nn.Module):
    def __init__(self, configs):
        super(base_model_blk1, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(configs.input_channels, 32, kernel_size=configs.kernel_size,
                      stride=configs.stride, bias=False, padding=(configs.kernel_size // 2)),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(configs.dropout)
        )

    def forward(self, x_in):
        x = self.conv_block1(x_in)
        return x


class base_model_blk2(nn.Module):
    def __init__(self, configs):
        super(base_model_blk2, self).__init__()

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3, padding=1),
            nn.Dropout(configs.dropout)
        )

    def forward(self, x_in):
        x = self.conv_block2(x_in)
        return x


class base_model_blk3(nn.Module):
    def __init__(self, configs):
        super(base_model_blk3, self).__init__()
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(64, configs.final_out_channels, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(configs.final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3, padding=1),
            nn.Dropout(configs.dropout)
        )

    def forward(self, x_in):
        x = self.conv_block3(x_in)
        return x



class cnn_feature_extractor(nn.Module):
    def __init__(self, configs):
        super(cnn_feature_extractor, self).__init__()
        self.conv_block1_shared = base_model_blk1(configs)
        self.conv_block2_shared = base_model_blk2(configs)
        self.conv_block3_shared = base_model_blk3(configs)

    def forward(self, input):
        out = self.conv_block1_shared(input)
        out = self.conv_block2_shared(out)
        out = self.conv_block3_shared(out)
        
        return out

class Classifier(nn.Module):
    def __init__(self, configs):
        super(Classifier, self).__init__()
        model_output_dim = configs.features_len
        self.logits = nn.Linear(model_output_dim * configs.final_out_channels, configs.num_classes)

    def forward(self, input):
        logits = self.logits(input)
        return logits


class Discriminator(nn.Module):
    def __init__(self, configs):
        super(Discriminator, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(configs.features_len * configs.final_out_channels, configs.disc_hid_dim),
            nn.ReLU(),
            nn.Linear(configs.disc_hid_dim, configs.disc_hid_dim),
            nn.ReLU(),
            nn.Linear(configs.disc_hid_dim, 1)
        )

    def forward(self, input):
        out = self.layer(input)
        return out



class Self_Attn(nn.Module):
    def __init__(self, in_dim):
        super(Self_Attn, self).__init__()
        self.query_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim//2, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim//2, kernel_size=1)
        self.value_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        m_batchsize, C, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = x.view(m_batchsize, -1, width)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width)
        out_flat = out.reshape(out.shape[0], -1)
        return out_flat

