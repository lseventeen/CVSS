import models

model_2d = {
"UNet",
"FR_UNet",
"AttU_Net",
"CSNet",
"UNet_Nested"

}

model_3d = {
"D3_UNet",
"D3_FR_UNet",
"PHTrans"

}

def build_model(config):
    if config.MODEL.TYPE in model_2d:
        return getattr(models, config.MODEL.TYPE)(
        num_classes = 1,
        num_channels = 8
        ), True
    elif config.MODEL.TYPE in model_3d:
        return getattr(models, config.MODEL.TYPE)(
        num_classes = 1,
        num_channels = 1
        ), False
    else:
        raise NotImplementedError(f"Unkown model: {config.MODEL.TYPE}")


 