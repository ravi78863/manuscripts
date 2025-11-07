import segmentation_models_pytorch as smp

def get_model(encoder_name='resnet34', encoder_weights='imagenet', in_channels=3, classes=2):
    '''
    Returns a U-Net model for manuscript damage segmentation.
    encoder_name: Backbone CNN (e.g. 'resnet34', 'efficientnet-b0')
    classes: Number of output channels (2 = physical + written)
    '''
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes
    )
    return model
conda install -c conda-forge opencv