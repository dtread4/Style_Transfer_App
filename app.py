# David Treadwell

import streamlit as st

import pandas as pd

from PIL import Image
import io

from gatys import gatys_style_transfer, INPUT_IMAGE_TYPES


VALID_FILE_TYPES = ['png', 'jpeg', 'tiff']


def stimage_to_pil(st_image):
    """
    Converts an image uploaded to Streamlit to a PIL image (more standard format for ML processing)

    Args:
        st_image: The streamlit uploaded image

    Returns:
        The PIL loaded image
    """
    image_bytes = st_image.read()
    image_buffer = io.BytesIO(image_bytes)
    pil_image = Image.open(image_buffer)
    return pil_image


def pil_to_bytes(pil_image, image_type='png'):
    """
    Converts a PIL image to a bytes image to save with Streamlit

    Args:
        pil_image: The PIL image to convert
        image_type: The image file type (e.g. png, jpg, etc.). Defaults to 'PNG'.

    Returns:
        The bytes transformed image
    """
    buffer = io.BytesIO()
    pil_image.save(buffer, format=image_type)
    byte_image = buffer.getvalue()
    return byte_image


# Button to display customization parameters
def toggle_state(state_key):
    """
    Callback function to swap state for buttons
    Since other parts of the UI depend on checking the boolean state, assumes that the state_key always exists

    Args:
        state_key: The state key name to toggle
    """
    st.session_state[state_key] = not st.session_state[state_key]


def process_filename(filename):
    """
    Removes the file type from a string for the filename

    Args:
        filename: The filename to process

    Returns:
        The filename without any file type extensions in it
    """
    # If any of the valid file types are in the string, remove them
    for extension in VALID_FILE_TYPES:
        full_extension = f".{extension}"
        if full_extension in filename:
            filename = filename.replace(full_extension, '')
    return filename


# ~~~~~~~~~~~~~~~~~~~~~~~ DEFAULT VALUES ~~~~~~~~~~~~~~~~~~~~~~~~~~~

TRANSFER_PARAMETERS_PREPEND = "transfer_parameters_"

DEFAULT_TRANSFER_PARAMETERS = {
    'input_type': 'Content image',
    'style_weight': 1000000, 
    'content_weight': 1,
    'num_steps': 300,
    'height': 256, 
    'width': 256,
    'device': 'auto', 
    'rng_seed': 42,
    'display_loss_steps': 50,
    'image_file_type': 'png',
    'display_loss': False
}

def reset_defaults_parameters():
    """
    Resets all of the parameters to be the default parameters
    """
    for key, value in DEFAULT_TRANSFER_PARAMETERS.items():
        st.session_state[TRANSFER_PARAMETERS_PREPEND + key] = value


# Create a deep copy of the default parameters and put them in the session state
if 'default_params_set' not in st.session_state:
    reset_defaults_parameters()
    st.session_state.default_params_set = True


# ~~~~~~~~~~~~~~~~~~~~~~~~~ START OF UI ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


content_col, style_col = st.columns(spec=2)

# Column 1 for content image upload
with content_col:
    content_image_stl = st.file_uploader("Upload Content Image", type=VALID_FILE_TYPES)
    if content_image_stl is not None:
        st.image(content_image_stl)

        # Convert uploaded image to PIL image
        content_image_pil = stimage_to_pil(content_image_stl)

# Column 2 for style image upload
with style_col:
    style_image_stl = st.file_uploader("Upload Style Image", type=VALID_FILE_TYPES)
    if style_image_stl is not None:
        st.image(style_image_stl)

        # Convert uploaded image to PIL image
        style_image_pil = stimage_to_pil(style_image_stl)

# Create a session state to store whether or not to display customization
if 'display_parameters' not in st.session_state:
    st.session_state.display_parameters = False

# Create a session state to store whether or not to display advanced customization
if 'display_advanced_params' not in st.session_state:
    st.session_state.display_advanced_params = False

# Change button text if options are displayed or not
param_button_text = "Display customization options" if not st.session_state.display_parameters else "Hide customization options"
st.button(param_button_text, on_click=toggle_state, args=('display_parameters',))

# Customization parameters
# TODO make default values appear properly when menus opened
if st.session_state.display_parameters:
    st.header("Customize Parameters for Style Transfer")
    param_col_1, param_col_2 = st.columns(spec=2)
    with param_col_1:
        st.selectbox(
            "Input image ('base' for style transfer). Default is content image",
            (input_img for input_img in INPUT_IMAGE_TYPES),
            index=0,
            key=f'{TRANSFER_PARAMETERS_PREPEND}input_type'
        )
        st.number_input(
            "Number of steps (iterations) to apply style transfer. More steps means more blending. Default is 300, range is [50, 1000].",
            min_value=50,
            max_value=1000,
            step=50,
            format="%d",
            value=st.session_state[f'{TRANSFER_PARAMETERS_PREPEND}num_steps'],
            key=f'{TRANSFER_PARAMETERS_PREPEND}num_steps'
        )
    with param_col_2:
        st.number_input(
            "Weight to apply to the style during transfer. Default is 1,000,000. Range is [10,000, 10,000,000]",
            min_value=100,
            max_value=10000000,
            step=250,
            format='%d',
            value=st.session_state[f'{TRANSFER_PARAMETERS_PREPEND}style_weight'],
            key=f'{TRANSFER_PARAMETERS_PREPEND}style_weight'
        )
        st.number_input(
            "Weight to apply to the content during transfer. Default is 1. Range is [1, 100]",
            min_value=1,
            max_value=100,
            step=1,
            format='%d',
            value=st.session_state[f'{TRANSFER_PARAMETERS_PREPEND}content_weight'],
            key=f'{TRANSFER_PARAMETERS_PREPEND}content_weight'
        )

    # Advanced options (requires button press to limit default available options)
    advanced_button_text = "Display advanced options" if not st.session_state.display_advanced_params else "Hide advanced options"
    st.button(advanced_button_text, on_click=toggle_state, args=('display_advanced_params',))
    
    # Advanced options
    if st.session_state.display_advanced_params:
        st.header("Advanced Customization options")
        adv_col_1, adv_col_2 = st.columns(spec=2)
        with adv_col_1:
            st.number_input(
                "Image height. Default is 512. Range is [32, 2048]",
                min_value=32,
                max_value=2048,
                step=1,
                format='%d',
                value=st.session_state[f'{TRANSFER_PARAMETERS_PREPEND}height'],
                key=f'{TRANSFER_PARAMETERS_PREPEND}height'
            )
            st.number_input(
                "Image width. Default is 512. Range is [32, 2048]",
                min_value=32,
                max_value=2048,
                step=1,
                format='%d',
                value=st.session_state[f'{TRANSFER_PARAMETERS_PREPEND}width'],
                key=f'{TRANSFER_PARAMETERS_PREPEND}width'
            )
            st.text_input(
                "Device to use. Defaults to 'auto'. Recommend to only use 'auto', which automatically selects CPU/GPU, unless you are an advanced PyTorch user.",
                value=st.session_state[f'{TRANSFER_PARAMETERS_PREPEND}device'],
                key=f'{TRANSFER_PARAMETERS_PREPEND}device'
            )
        with adv_col_2:
            st.number_input(
                "Random seed to use for training. Default is 42. Range is [0, 1000] (for simplicity).",
                min_value=0,
                max_value=1000,
                step=1,
                format='%d',
                value=st.session_state[f'{TRANSFER_PARAMETERS_PREPEND}rng_seed'],
                key=f'{TRANSFER_PARAMETERS_PREPEND}rng_seed'
            )
            st.checkbox(  # TODO implement the loss display
                'Display loss at training steps?',
                key=f'{TRANSFER_PARAMETERS_PREPEND}display_loss'
            )
            if st.session_state[f'{TRANSFER_PARAMETERS_PREPEND}display_loss']:
                st.number_input(
                    "Step interval to display loss at. Default is 50. Range is [1, number of steps]",
                    min_value=1,
                    max_value=st.session_state[TRANSFER_PARAMETERS_PREPEND + 'num_steps'],
                    step=1,
                    format='%d',
                    value=st.session_state[f'{TRANSFER_PARAMETERS_PREPEND}display_loss_steps'],
                    key=f'{TRANSFER_PARAMETERS_PREPEND}display_loss_steps'
                )
    
    # Button to reset the parameters to default
    back_to_default_params = st.button("Reset parameters to defaults", on_click=reset_defaults_parameters)
    if back_to_default_params:
        st.rerun()  # Clears the note about style transfer running

# Show a button to run style transfer once both images uploaded
style_transferred_image_pil = None
if content_image_stl is not None and style_image_stl is not None:

    # When the button is clicked, run the program
    if st.button("Run style transfer"):
        st.header("Transferring style image's style to content image...") # TODO make this a progress bar
        style_transferred_image_pil = gatys_style_transfer(
            content_image_pil=content_image_pil, 
            style_image_pil=style_image_pil, 
            input_type=st.session_state[TRANSFER_PARAMETERS_PREPEND + 'input_type'],
            style_weight=st.session_state[TRANSFER_PARAMETERS_PREPEND + 'style_weight'], 
            content_weight=st.session_state[TRANSFER_PARAMETERS_PREPEND + 'content_weight'],
            num_steps=st.session_state[TRANSFER_PARAMETERS_PREPEND + 'num_steps'],
            height=st.session_state[TRANSFER_PARAMETERS_PREPEND + 'height'], 
            width=st.session_state[TRANSFER_PARAMETERS_PREPEND + 'width'],
            device=st.session_state[TRANSFER_PARAMETERS_PREPEND + 'device'], 
            rng_seed=st.session_state[TRANSFER_PARAMETERS_PREPEND + 'rng_seed'],
            display_loss_steps=st.session_state[TRANSFER_PARAMETERS_PREPEND + 'display_loss_steps']
        )  # TODO Add option to display run losses (and a graph?)

        # Put the bytes image into the session state so that it persists if the user downloads the image
        st.session_state.pil_image = style_transferred_image_pil
        
        st.rerun()  # Clears the note about style transfer running

# Message before content and style images are displayed
else:
    st.header("Upload a content and style image to run style transfer with")


# Display the style transferred image with a download button
if 'pil_image' in st.session_state:
    # Set default name and file type
    if 'image_file_name' not in st.session_state:
        st.session_state.image_file_name = 'style_transfer_output'
    if 'image_file_type' not in st.session_state:
        st.session_state.image_file_type = 'png'
    image_file_name = f"{st.session_state['image_file_name']}.{st.session_state['image_file_type']}"
    
    # Get the bytes image
    st.session_state.bytes_image = pil_to_bytes(st.session_state.pil_image, 
                                                st.session_state['image_file_type'])

    # Displays the image
    st.image(st.session_state.bytes_image, caption='Style transferred image')

    # Provide options for file name and extension
    name_col, type_col = st.columns(spec=2)
    with name_col:
        process_filename(st.text_input(
            "File name for downloaded image (file types will be ignored).",
            key='image_file_name'  
        ))
    with type_col:
        st.selectbox(
            "File type to save as. Default is .png",
            (extension for extension in VALID_FILE_TYPES),
            index=0,  # Default is png #
            key='image_file_type'
        )
    
    # Update session image in case type changes from user selection
    download_image = pil_to_bytes(st.session_state.pil_image, 
                                    image_type=st.session_state['image_file_type'])


    # Allows for download
    st.download_button(
        label='Download style-transferred image',
        data=download_image,
        file_name=image_file_name
    )
