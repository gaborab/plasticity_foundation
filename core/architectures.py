import torch.nn as nn
from timm.models import create_model
from core import settings
from libs.labram import modeling_finetune


class EEGAutoencoder(nn.Module):
    def __init__(self, target_dim):
        super(EEGAutoencoder, self).__init__()
        self.model = create_model(
            settings.LABRAM_VERSION,
            pretrained=False,
        )
        input_dim = self.model.embed_dim
        hidden_dim = input_dim // 2
        # set model head as an MLP for autoencoding
        self.model.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, target_dim),
        )

    def forward(self, x, input_chans):
        return self.model(x, input_chans)

    def get_num_layers(self):
        return self.model.get_num_layers()

    def no_weight_decay(self):
        return self.model.no_weight_decay()
