
import torch
from torch import nn
import numpy as np

from biom3d.bmz.models.encoder_vgg import EncoderBlock, VGGEncoder
from biom3d.bmz.models.decoder_vgg_deep import VGGDecoder

#---------------------------------------------------------------------------
# 3D UNet with the previous encoder and decoder

class UNet(nn.Module):
    def __init__(
        self, 
        num_pools=[5,5,5], 
        num_classes=1, 
        factor=32,
        encoder_ckpt = None,
        model_ckpt = None,
        use_deep=True,
        in_planes = 1,
        flip_strides = False,
        roll_strides = True, #used for models trained before commit f2ac9ee (August 2023)
        ):
        super(UNet, self).__init__()
        self.encoder = VGGEncoder(
            EncoderBlock,
            num_pools=num_pools,
            factor=factor,
            in_planes=in_planes,
            flip_strides=flip_strides,
            roll_strides=roll_strides,
            )
        self.decoder = VGGDecoder(
            EncoderBlock,
            num_pools=num_pools,
            num_classes=num_classes,
            factor_e=factor,
            factor_d=factor,
            use_deep=use_deep,
            flip_strides=flip_strides,
            roll_strides=roll_strides,
            )

        # load encoder if needed
        if encoder_ckpt is not None:
            print("Load encoder weights from", encoder_ckpt)
            if torch.cuda.is_available():
                self.encoder.cuda()
            ckpt = torch.load(encoder_ckpt)
            if 'model' in ckpt.keys():
                # remove `module.` prefix
                state_dict = {k.replace("module.", ""): v for k, v in ckpt['model'].items()} 
                # remove `0.` prefix induced by the sequential wrapper
                state_dict = {k.replace("0.layers", "layers"): v for k, v in state_dict.items()} 
                # remove `backbone.` prefix induced by pretraining 
                state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
                print(self.encoder.load_state_dict(state_dict, strict=False))
            elif 'teacher' in ckpt.keys():
                # remove `module.` prefix
                state_dict = {k.replace("module.", ""): v for k, v in ckpt['teacher'].items()}  
                # remove `backbone.` prefix induced by multicrop wrapper
                state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
                print(self.encoder.load_state_dict(state_dict, strict=False))
            else:
                print("[Warning] the following encoder couldn't be loaded, wrong key:", encoder_ckpt)
        
        if model_ckpt is not None:
            self.load(model_ckpt)

    def freeze_encoder(self, freeze=True):
        """
        freeze or unfreeze encoder model
        """
        if freeze:
            print("Freezing encoder weights...")
        else:
            print("Unfreezing encoder weights...")
        for l in self.encoder.parameters(): 
            l.requires_grad = not freeze
    
    def unfreeze_encoder(self):
        self.freeze_encoder(False)

    def load(self, model_ckpt):
        """Load the model from checkpoint.
        The checkpoint dictionary must have a 'model' key with the saved model for value.
        """
        print("Load model weights from", model_ckpt)
        if torch.cuda.is_available():
            self.cuda()
            device = torch.device('cuda')
        else:
            self.cpu()
            device = torch.device('cpu')
        ckpt = torch.load(model_ckpt, map_location=device)
        if 'encoder.last_layer.weight' in ckpt['model'].keys():
            del ckpt['model']['encoder.last_layer.weight']
        print(self.load_state_dict(ckpt['model'], strict=False))

    def forward(self, x): 
        # x is an image
        out = self.encoder(x)
        out = self.decoder(out)
        return out[0]

#---------------------------------------------------------------------------