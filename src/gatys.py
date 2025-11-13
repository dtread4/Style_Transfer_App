# David Treadwell
#
# Based on the work of Gatys et al. "A Neural Algorithm of Artistic Style"
# https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf
#
# Adapted from PyTorch tutorial on Neural Transfer
# https://docs.pytorch.org/tutorials/advanced/neural_style_tutorial.html

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
from torchvision.models import vgg19, VGG19_Weights

import time
import random

import warnings
warnings.filterwarnings("ignore", message='To copy construct from a tensor, it is recommended to use sourceTensor.detach().clone() or sourceTensor.detach().clone().requires_grad_(True), rather than torch.tensor(sourceTensor).')


# Standard VGG-19 network
VGG_19 = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()

# Standard content and style layers of the VGG-19 network used to apply the style transfer
CONTENT_LAYERS_DEFAULT = ['conv_4']
STYLE_LAYERS_DEFAULT = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

# VGG default means and standard deviations
VGG_NORM_MEAN_DEFAULT = torch.tensor([0.485, 0.456, 0.406])
VGG_NORM_STD_DEFAULT = torch.tensor([0.229, 0.224, 0.225])

# Valid input image types
INPUT_IMAGE_TYPES = ["Content image", "Style image", "White noise image"]


def set_torch_determinism(seed):
    """
    Sets universal determinism

    Args:
        seed: The random seed used to ensure reproducibility (should be an integer)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_device(device_name='auto'):
    """
    Sets the device to use when training or running model inference

    Args:
        device_name: The name of the device to use (should be 'cpu', 'cuda', or 'auto')
                     'auto' is the default, and will select 'cuda' if available or 'cpu' if not
                     'cuda' may specify device number, like 'cuda:0'

    Returns:
        The name of the device being used
    """
    # Set device if designated, or default to cuda if not specified
    if device_name != 'auto':
        device = torch.device(device_name)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Performing style transfer with device: {device.type}")

    # Set the device    
    torch.set_default_device(device)
    return device


def set_image_size(device):
    """
    Sets the size of the images to perform style transfer with based on the device being used
    GPU (cuda) = (512, 512)
    CPU = (128, 128)
    Smaller images will be transferred faster when using the CPU

    Args:
        device: The device being used to perform style transfer

    Returns:
        The integer size of the image
    """
    return (512, 512) if device.type == 'cuda' else (128, 128)


def prepare_image(image, image_size):
    """
    Loads an image with transforms applied

    Args:
        image: The image to transform for style transfer
        image_size: The integer size of the image

    Returns:
        The image as a tensor ready for use
    """
    # Create a set of standard transforms to load the image with. Requires knowledge of image size
    image_transforms = transforms.Compose(
        [transforms.Resize(image_size),  # Scale imported image to correct size
         transforms.ToTensor()         # Transform the image to a torch tensor
         ]
    )

    # Load the image then apply transforms
    transformed_image = image_transforms(image).unsqueeze(0)  # Add batch dimension
    transformed_image = transformed_image.to(torch.get_default_device())

    return transformed_image


def convert_tensor_to_pil_image(tensor_image):
    """
    Converts an image in tensor form to a PIL image

    Args:
        tensor_image: The image as a tensor (shape [1, c, h, w])

    Returns:
        The image as a PIL image
    """
    image_display = tensor_image.cpu().clone()  # Display without changing actual image
    image_display = image_display.squeeze()  # Remove batch dimension
    to_pil_image = transforms.ToPILImage()
    image_display = to_pil_image(image_display)
    return image_display


def display_image(tensor_image, title=None):
    """
    Displays a tensor image

    Args:
        tensor_image: _description_
        title: _description_. Defaults to None.
    """
    plt.ion()  # Interactive display updates

    # Convert a copy of the image to PIL for display
    image_display = convert_tensor_to_pil_image(tensor_image)

    plt.imshow(image_display)
    if title is not None:
        plt.title(title)
    plt.axis('off')
    plt.pause(0.001)  # pausing so plots are updated properly


class ContentLoss(nn.Module):
    """
    Used to calculate content loss.
    Note that this is not a true loss function - it computes loss to calculate gradients automatically,
    but operates in a transparent manner
    """

    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()

    def forward(self, input):
        """
        Forward method of Gatys model to compute gradients at content loss layers

        Args:
            input: The current image that has style transfer performed on it

        Returns:
            The original input, making the layer transparent
        """
        self.loss = F.mse_loss(input, self.target)  # self.target is the content loss image
        return input
    

class StyleLoss(nn.Module):
    """
    Used to calculate style loss.
    Note that this is not a true loss function - it computes loss to calculate gradients automatically,
    but operates in a transparent manner
    """

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = self.gram_matrix(target_feature).detach()

    def forward(self, input):
        """
        Forward method of Gatys model to compute gradients at style loss layers

        Args:
            input: The current input image that has style transfer performed on it

        Returns:
            The original input, making the layer transparent
        """
        G = self.gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)  # self.target is the style loss image
        return input
    
    def gram_matrix(self, input):
        """
        Calculates the Gram matrix for an image (tensor matrix) - result of multiplying matrix by its transposed matrix

        Args:
            input: The input tensor image to calculate the Gram matrix of

        Returns:
            The normalized Gram matrix
        """
        # a=batch size(=1)
        # b=number of feature maps
        # (c,d)=dimensions of a feature map (N=c*d)
        a, b, c, d = input.size()  

        # resize F_XL into \hat F_XL
        features = input.view(a * b, c * d)  

        # compute the gram product
        G = torch.mm(features, features.t())  

        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return G.div(a * b * c * d)
    

class Normalization(nn.Module):
    """
    Module to normalize images for use with a nn.Sequential network
    """

    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = mean.view(-1, 1, 1)
        self.std = std.view(-1, 1, 1)

    def forward(self, image):
        """
        Forward pass to normalize image (subtract mean then divide by standard deviation)

        Args:
            image: The image to normalize

        Returns:
            The normalized image
        """
        return (image - self.mean) / self.std
    

def get_model_and_losses(style_image, content_image,
                         cnn=VGG_19,
                         normalization_mean=VGG_NORM_MEAN_DEFAULT, 
                         normalization_std=VGG_NORM_STD_DEFAULT,
                         content_layers=CONTENT_LAYERS_DEFAULT,
                         style_layers=STYLE_LAYERS_DEFAULT):
    """
    Creates the model for style transfer

    Args:
        style_image: Style image to transfer onto the content image
        content_image: Content image to have its style transferred from the style image
        cnn: The CNN to use for style transfer (should be VGG-19)
        normalization_mean: Image means to normalize with. Defaults to VGG_NORM_MEAN_DEFAULT.
        normalization_std: Image standard deviations. Defaults to VGG_NORM_STD_DEFAULT.
        content_layers: Layers to calculate content gradients with. Defaults to CONTENT_LAYERS_DEFAULT.
        style_layers: Layers to calculate style gradients with. Defaults to STYLE_LAYERS_DEFAULT.

    Raises:
        RuntimeError: When unknown CNN layers are reached, an error can be raised. 
                      This network expects the following layers (and NO other ones):
                      - Conv2d (convolutional layer)
                      - ReLU (activation layer)
                      - MaxPool2d (pooling layer)
                      - BatchNorm2d (batch normalization layer)

    Returns:
        The model with only style and content layers included, and the style and content losses
    """
    # Ensure cnn and mean/std are on correct device
    cnn = cnn.to(torch.get_default_device())
    normalization_mean = normalization_mean.to(torch.get_default_device())
    normalization_std = normalization_std.to(torch.get_default_device())

    # Create normalization module
    normalization = Normalization(normalization_mean, normalization_std)

    # Storage to have an iterable access to or list of content/style losses
    content_losses = []
    style_losses = []

    # assuming that ``cnn`` is a ``nn.Sequential``, so we make a new ``nn.Sequential``
    # to put in modules that are supposed to be activated sequentially
    # E.g. include the style and content loss in the correct places
    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv (layer), so we can easily track what layer we are on
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ``ContentLoss``
            # and ``StyleLoss`` we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        # Add content loss
        if name in content_layers:
            target = model(content_image).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        # Add style loss:
        if name in style_layers:
            target_feature = model(style_image).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # Remove layers after the last content and style loss layers
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses


def create_white_noise_image(image_size):
    """
    Creates a white noise image of the specified size

    Args:
        image_size: The size of the white noise image to create
                    Note that the size should be the shape of [1, c, h, w]
                    1 is the pseudo-batch size
                    c is the number of channels (typically 3, like RGB)
                    h, w are height and width

    Returns:
        The white noise image (as a tensor)
    """
    return torch.randn(image_size)


def get_input_optimizer(input_image):
    """
    Creates an L-BFGS algorithm to run gradient descent to train the image, rather than to train the network

    Args:
        input_image: The image we are training to mimic the style/content of the transferring images

    Returns:
        The optimizer parameter, which requires a gradient
    """
    optimizer = optim.LBFGS([input_image])
    return optimizer


def run_style_transfer(input_image, style_image, content_image,
                       cnn=VGG_19,
                       normalization_mean=VGG_NORM_MEAN_DEFAULT, 
                       normalization_std=VGG_NORM_STD_DEFAULT,
                       num_steps=300,
                       style_weight=1000000, 
                       content_weight=1):
    """
    Runs the style transfer process using the specified images and network

    Args:
        input_image: The image to transfer the style and content blend onto
        style_image: The image to take the style from
        content_image: The image to take the content fromm
        cnn: CNN used to calculate gradients for style transfer. Defaults to VGG_19.
        normalization_mean: Means to normalize images with. Defaults to VGG_NORM_MEAN_DEFAULT.
        normalization_std: Standard deviations to normalize images. Defaults to VGG_NORM_STD_DEFAULT.
        num_steps: How many steps to perform style transfer with. Defaults to 300.
        style_weight: Weight to give to style image. Defaults to 1000000.
        content_weight: Weight to give to content image. Defaults to 1.

    Returns:
        The style-transferred image (as a tensor)
    """
    
    # Initialize the style transfer model
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_model_and_losses(style_image=style_image, content_image=content_image,
                                                               cnn=cnn, 
                                                               normalization_mean=normalization_mean,
                                                               normalization_std=normalization_std)

    # We want to optimize the input and not the model parameters so we update all the requires_grad fields accordingly
    input_image.requires_grad_(True)

    # We also put the model in evaluation mode, so that specific layers 
    # such as dropout or batch normalization layers behave correctly.
    model.eval()
    model.requires_grad_(False)

    # Get the optimizer parameter
    optimizer = get_input_optimizer(input_image)

    # Trim the input image to match the style and content of the corresponding images
    print('Optimizing...')
    run = [0]
    start_time = int(round(time.time()))
    while run[0] <= num_steps:

        # Adjust values of input image to train it
        def closure():
            # Ensure that updated values are in the range [0, 1]
            with torch.no_grad():
                input_image.clamp_(0, 1)

            # Reset optimizer for current step
            optimizer.zero_grad()
            model(input_image)
            style_score = 0
            content_score = 0

            # Aggregate (sum) style and content losses
            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            # Adjust style and content losses by corresponding weights
            style_score *= style_weight
            content_score *= content_weight

            # Combine style and content losses and perform backpropagation
            loss = style_score + content_score
            loss.backward()

            # Print diagnostics every 50 steps
            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()

            return style_score + content_score

        optimizer.step(closure)

    # Final correction to ensure output style-transferred image is in the range [0, 1]
    with torch.no_grad():
        input_image.clamp_(0, 1)

    # print concluding message
    end_time = int(round(time.time()))
    print(f"Finished style transfer! Took {end_time - start_time} seconds.")

    return input_image


def save_image(tensor_image, output_path):
    """
    Saves an image to a specified output path

    Args:
        tensor_image: The image to save as a tensor (shape [1, c, h, w])
        output_path: _description_
    """
    image = convert_tensor_to_pil_image(tensor_image)
    image.save(output_path)


def set_input_image(content_image, style_image, input_type):
    """
    Creates the proper input image - the image that will have style transfer applied to it
    Will either create a white noise image with the proper dimensions, or a deep copy of the content or style image

    Args:
        content_image: The content image in style transfer
        style_image: The style image in style transfer
        input_type: What to return for the input image

    Returns:
        The designated input image
    """
    if input_type == INPUT_IMAGE_TYPES[2]:
        # Use content size to also include batch and channel dimensions
        return create_white_noise_image(content_image.size())
    if input_type == INPUT_IMAGE_TYPES[0]:
        return content_image.clone()
    if input_type == INPUT_IMAGE_TYPES[1]:
        return style_image.clone()


def gatys_style_transfer(content_image_pil, 
                         style_image_pil, 
                         input_type='Content image',
                         style_weight=1000000, 
                         content_weight=1,
                         num_steps=300,
                         height=512, 
                         width=512,
                         device='auto', 
                         rng_seed=42,
                         display_loss_steps=50):  # TODO implement display steps
    """
    Performs the Gatys method for style transfer

    Args:
        content_image_pil: The PIL pre-loaded content image
        style_image_pil: The PIL pre-loaded style image
        input_type: The type of input image to use (content, style, white noise). Defaults to 'content'.
        style_weight: The weight given to the style image. Typically higher than content. Defaults to 1000000.
        content_weight: The weight given to the content image. Typically lower than style. Defaults to 1.
        num_steps: How many steps to train style transfer for. Defaults to 300.
        height: The height to use for the images (must be same for all images, so set here). Defaults to 512.
        width: The width to use for the images (must be same for all images, so set here). Defaults to 512.
        device: The device to use to perform style transfer. Defaults to 'auto'.
        rng_seed: Seed to use for RNG (maintains determinism and reproducibility). Defaults to 42.
        display_loss_steps: How often to display loss values for style and content. Defaults to 50.

    Returns:
        The style-transferred output image
    """
    # Print the values being used
    print("\n--- Values for style transfer ---")
    for name, value in locals().items():
        if name != 'content_image_pil' and name != 'style_image_pil':
            print(f"{name}: {value}")
    print("---------------------------------\n")

    # Make the process deterministic
    set_torch_determinism(seed=rng_seed)

    # Set the device
    device = set_device(device_name=device)
    image_size = (height, width)

    # Prepare the content and style images for transfer
    content_image_prepared = prepare_image(content_image_pil, image_size)
    style_image_prepared = prepare_image(style_image_pil, image_size)

    # Use the correct input image
    input_image = set_input_image(content_image_prepared, style_image_prepared, input_type)

    # Perform style transfer
    output = run_style_transfer(input_image=input_image,
                                style_image=style_image_prepared,
                                content_image=content_image_prepared,
                                num_steps=num_steps,
                                style_weight=style_weight, content_weight=content_weight)
    
    # Return the output as a PIL image
    return convert_tensor_to_pil_image(output)
