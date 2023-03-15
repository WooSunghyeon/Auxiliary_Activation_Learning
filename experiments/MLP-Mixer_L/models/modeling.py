import copy

import torch

import models.configs as configs

from os.path import join as pjoin

from torch import nn
from torch.nn.modules.utils import _pair
import aal.aal as aal



TOK_FC_0 = "token_mixing/Dense_0"
TOK_FC_1 = "token_mixing/Dense_1"
CHA_FC_0 = "channel_mixing/Dense_0"
CHA_FC_1 = "channel_mixing/Dense_1"
PRE_NORM = "LayerNorm_0"
POST_NORM = "LayerNorm_1"


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


class MlpBlock(nn.Module):
    def __init__(self, hidden_dim, ff_dim, mode):
        super(MlpBlock, self).__init__()
        if mode == 0:
            self.fc0 = nn.Linear(hidden_dim, ff_dim, bias=True)
            self.fc1 = nn.Linear(ff_dim, hidden_dim, bias=True)
        elif mode == 1:
            self.fc0 = nn.Linear(hidden_dim, ff_dim, bias=True)
            self.fc1 = aal.Linear_ASA(ff_dim, hidden_dim, bias=True)
        else:
            self.fc0 = aal.Linear_ASA(hidden_dim, ff_dim, bias=True)
            self.fc1 = aal.Linear_ASA(ff_dim, hidden_dim, bias=True)
        
        self.act_fn = nn.GELU()

    def forward(self, x):
        x = self.fc0(x)
        x = self.act_fn(x)
        x = self.fc1(x)
        return x


class MixerBlock(nn.Module):
    def __init__(self, config, learning_rule):
        super(MixerBlock, self).__init__()
        if learning_rule == 'bp':
            mode0, mode1 = 0,0
        elif learning_rule == 'asa1':
            mode0, mode1 = 0,1
        elif learning_rule == 'asa2':
            mode0, mode1 = 0,2
        elif learning_rule == 'asa3':
            mode0, mode1 = 1,2
        elif learning_rule == 'asa4':
            mode0, mode1 = 2,2
        self.token_mlp_block = MlpBlock(config.n_patches, config.tokens_mlp_dim, mode=mode0)
        self.channel_mlp_block = MlpBlock(config.hidden_dim, config.channels_mlp_dim, mode=mode1)
        self.pre_norm = nn.LayerNorm(config.hidden_dim, eps=1e-6)
        self.post_norm = nn.LayerNorm(config.hidden_dim, eps=1e-6)

    def forward(self, x):
        h = x
        x = self.pre_norm(x)
        x = x.transpose(-1, -2)
        x = self.token_mlp_block(x)
        x = x.transpose(-1, -2)
        x = x + h

        h = x
        x = self.post_norm(x)
        x = self.channel_mlp_block(x)
        x = x + h
        return x

    def load_from(self, weights, n_block):
        ROOT = f"MixerBlock_{n_block}"
        with torch.no_grad():
            self.token_mlp_block.fc0.weight.copy_(
                np2th(weights[pjoin(ROOT, TOK_FC_0, "kernel")]).t())
            self.token_mlp_block.fc1.weight.copy_(
                np2th(weights[pjoin(ROOT, TOK_FC_1, "kernel")]).t())
            self.token_mlp_block.fc0.bias.copy_(
                np2th(weights[pjoin(ROOT, TOK_FC_0, "bias")]).t())
            self.token_mlp_block.fc1.bias.copy_(
                np2th(weights[pjoin(ROOT, TOK_FC_1, "bias")]).t())

            self.channel_mlp_block.fc0.weight.copy_(
                np2th(weights[pjoin(ROOT, CHA_FC_0, "kernel")]).t())
            self.channel_mlp_block.fc1.weight.copy_(
                np2th(weights[pjoin(ROOT, CHA_FC_1, "kernel")]).t())
            self.channel_mlp_block.fc0.bias.copy_(
                np2th(weights[pjoin(ROOT, CHA_FC_0, "bias")]).t())
            self.channel_mlp_block.fc1.bias.copy_(
                np2th(weights[pjoin(ROOT, CHA_FC_1, "bias")]).t())

            self.pre_norm.weight.copy_(np2th(weights[pjoin(ROOT, PRE_NORM, "scale")]))
            self.pre_norm.bias.copy_(np2th(weights[pjoin(ROOT, PRE_NORM, "bias")]))
            self.post_norm.weight.copy_(np2th(weights[pjoin(ROOT, POST_NORM, "scale")]))
            self.post_norm.bias.copy_(np2th(weights[pjoin(ROOT, POST_NORM, "bias")]))


class MlpMixer(nn.Module):
    def __init__(self, config, img_size=224, num_classes=1000, patch_size=16, zero_head=False, learning_rule='bp'):
        super(MlpMixer, self).__init__()
        self.zero_head = zero_head
        self.num_classes = num_classes
        patch_size = _pair(patch_size)
        n_patches = (img_size // patch_size[0]) * (img_size // patch_size[1])
        config.n_patches = n_patches

        self.stem = nn.Conv2d(in_channels=3,
                              out_channels=config.hidden_dim,
                              kernel_size=patch_size,
                              stride=patch_size)
        self.head = nn.Linear(config.hidden_dim, num_classes, bias=True)
        self.pre_head_ln = nn.LayerNorm(config.hidden_dim, eps=1e-6)


        self.layer = nn.ModuleList()
        for _ in range(config.num_blocks):
            layer = MixerBlock(config, learning_rule)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, x, labels=None):
        x = self.stem(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)

        for block in self.layer:
            x = block(x)
        x = self.pre_head_ln(x)
        x = torch.mean(x, dim=1)
        logits = self.head(x)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
            return loss
        else:
            return logits

    def load_from(self, weights):
        with torch.no_grad():
            if self.zero_head:
                nn.init.zeros_(self.head.weight)
                nn.init.zeros_(self.head.bias)
            else:
                self.head.weight.copy_(np2th(weights["head/kernel"]).t())
                self.head.bias.copy_(np2th(weights["head/bias"]).t())
            self.stem.weight.copy_(np2th(weights["stem/kernel"], conv=True))
            self.stem.bias.copy_(np2th(weights["stem/bias"]))
            self.pre_head_ln.weight.copy_(np2th(weights["pre_head_layer_norm/scale"]))
            self.pre_head_ln.bias.copy_(np2th(weights["pre_head_layer_norm/bias"]))

            for bname, block in self.layer.named_children():
                block.load_from(weights, n_block=bname)


CONFIGS = {
    'Mixer-B_16': configs.get_mixer_b16_config(),
    'Mixer-L_16': configs.get_mixer_l16_config(),
    'Mixer-B_16-21k': configs.get_mixer_b16_config(),
    'Mixer-L_16-21k': configs.get_mixer_l16_config()
}
