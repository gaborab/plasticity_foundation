import torch.nn as nn
from timm.models import create_model
from einops import rearrange
from libs.labram import modeling_finetune # this import is needed for model to work

class BaseMLPWrapper(nn.Module):
    """
    Parent class for models with an MLP head.
    """
    def __init__(self, target_dim, hidden_dim, model_version='labram_base_patch200_200', pretrained=True):
        super().__init__()

        self.model = create_model(model_version, pretrained=pretrained)

        input_dim = self.model.embed_dim
        self.model.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, target_dim),
        )

    def get_num_layers(self):
        return self.model.get_num_layers()

    def no_weight_decay(self):
        return self.model.no_weight_decay()


class EEGAutoencoder(BaseMLPWrapper):
    """
    Inherits from BaseMLPWrapper to construct a model with an MLP head.
    The head is designed for autoencoding, where the target is the original input EEG epoch tensor.
    """
    def __init__(self, n_chans, target_dim, model_version='labram_base_patch200_200', pretrained=True, **kwargs):
        super(EEGAutoencoder, self).__init__(target_dim=target_dim, model_version=model_version, hidden_dim=512, pretrained=pretrained)
        self.n_chans = n_chans

    def forward(self, x, input_chans):
        pred = self.model(x, input_chans, return_patch_tokens=True)
        pred = rearrange(pred, 'B N T -> B (N T)')
        return pred

    def get_num_layers(self):
        return self.model.get_num_layers()

    def no_weight_decay(self):
        return self.model.no_weight_decay()

    def get_unflattened_output(self, x, input_chans = None):
        """
        Returns the predicted temporal output for the first channel, reshaped to (n_chans, time_points).
        """
        pred = self.model(x, input_chans, return_patch_tokens=True)
        if pred.ndim == 2:
            pred = rearrange(pred, 'B (N T) -> B N T', N=self.n_chans)
        return pred


class EEGSpectralDecoder(BaseMLPWrapper):
    """
    Inherits from BaseMLPWrapper to construct a model with an MLP head.
    The head is designed for spectral decoding, where the target is a spectral representation of the input EEG epoch tensor (e.g., wavelet coefficients).
    """
    def __init__(self, n_chans, n_freqs, target_dim, model_version='labram_base_patch200_200', pretrained=True, **kwargs):
        super(EEGSpectralDecoder, self).__init__(target_dim=target_dim, model_version=model_version, hidden_dim=1024, pretrained=pretrained)
        self.n_chans = n_chans
        self.n_freqs = n_freqs

    def forward(self, x, input_chans):
        pred = self.model(x, input_chans, return_patch_tokens=True)
        pred = rearrange(pred, 'B N W -> B (N W)')
        return pred

    def get_num_layers(self):
        return self.model.get_num_layers()

    def no_weight_decay(self):
        return self.model.no_weight_decay()

    def get_unflattened_output(self, x, input_chans = None):
        """
        Predicts spectral output for all channels, reshaped to (n_chans, n_freqs, time_points).
        """
        pred = self.model(x, input_chans, return_patch_tokens=True)

        pred = rearrange(pred, 'B N (F T) -> B N F T', F=self.n_freqs, N=self.n_chans)
        return pred
