from .d2.unet import UNet
import models

model_2d = {
"UNet",
"FR_UNet"

}

model_3d = {


}

def build_model(config):
    if config.MODEL.TYPE in model_2d:
        return getattr(models, config.MODEL.TYPE)(
        num_classes = 1,
        num_channels = 8
        )
    elif config.MODEL.TYPE in model_3d:
        return getattr(models, config.MODEL.TYPE)(
        num_classes = 1,
        num_channels = 1
        )


 