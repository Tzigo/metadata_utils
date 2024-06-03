import gradio as gr
import modules.sd_models as models
from modules.ui import create_refresh_button
from extensions.metadata_utils.scripts.metadata_util_lib import write_metadata
import os

import re
import json

from modules import script_callbacks


def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as ui_component:
        with gr.Tab("Adder"):
            with gr.Row():
                input_file = gr.Dropdown(models.checkpoint_tiles(), label="Checkpoint")
                create_refresh_button(input_file, models.list_models,
                                      lambda: {"choices": models.checkpoint_tiles()}, "refresh")

            with gr.Row():
                new_name = gr.Textbox(
                    placeholder='(Optional) Enter new checkpoint name. If omitted, appends "_md" to name',
                    max_lines=1, elem_id="new_name", label="New Name")

            with gr.Row():
                json_input = gr.Textbox(placeholder='Input JSON content', max_lines=10, elem_id="json_input",
                                        label="Metadata as JSON (SHIFT + SPACE = new line)")

            with gr.Row():
                gr.Column(scale=1)
                with gr.Column(scale=1):
                    button = gr.Button(value="Add Metadata", variant="primary")
                gr.Column(scale=1)

            button.click(on_button, inputs=[input_file, new_name, json_input])
        with gr.Tab("Reader"):
            with gr.Row():
                input_file = gr.Dropdown(models.checkpoint_tiles(), label="Checkpoint")
                create_refresh_button(input_file, models.list_models,
                                      lambda: {"choices": models.checkpoint_tiles()}, "refresh")
                load_metadata_button = gr.Button(value="load metadata", variant='primary')
            with gr.Row():
                metadata = gr.TextArea(label="Metadata")

            load_metadata_button.click(
                fn=on_button_load_metadata,
                inputs=[input_file],
                outputs=[metadata]
            )

    return [(ui_component, "Metadata Utils", "metadata_utils_tab")]


script_callbacks.on_ui_tabs(on_ui_tabs)


def get_model_path():
    # Forge support
    # get checkpoint directory from user config batch file
    cwd = os.getcwd()
    if 'Forge' in cwd:
        webui_bat_path = cwd + '\\' + 'webui-user.bat'

        # read in user batch config
        with open(webui_bat_path, 'r', encoding='utf-8') as file:
            text = file.read()

        # extract directory from flag
        start_index = text.find('--ckpt-dir "') + len('--ckpt-dir "')
        end_index = text.find('"', start_index)
        if start_index != -1 and end_index != -1:
            return text[start_index:end_index]
        else:
            gr.Warning('Please check your Forge ckpt-dir flag in "webui-user.bat"!')
            return
    else:
        return models.model_path


def on_button_load_metadata(input_file: str):
    # get model path for used ui version
    model_path = get_model_path()

    # Stop if no checkpoint is selected
    if len(input_file) == 0:
        gr.Warning("Please select a checkpoint")
        return

    # Extract file name from Dropdown value
    file_name = re.findall(r".+\.safetensors", input_file)

    # Stop if no valid safetensors file is found in Dropdown value
    if len(file_name) == 0:
        gr.Warning("Selected model could not be processed or is not a .safetensors file")
        return

    # Build path from model path and first extracted file name
    file_path = str(os.path.join(model_path, file_name[0]))

    try:
        loaded_metadata = models.read_metadata_from_safetensors(file_path)
    except AssertionError:
        return "Checkpoint is not a .safetensors file"

    if not loaded_metadata:
        return "No metadata"
    return json.dumps(loaded_metadata, indent=4)


def on_button(input_file: str, new_name: str, json_input: str):
    # get model path for used ui version
    model_path = get_model_path()

    # Stop if no checkpoint is selected
    if len(input_file) == 0:
        gr.Warning("Please select a checkpoint")
        return

    # Extract file name from Dropdown value
    file_name = re.findall(r".+\.safetensors", input_file)

    # Stop if no valid safetensors file is found in Dropdown value
    if len(file_name) == 0:
        gr.Warning("Selected model could not be processed or is not a .safetensors file")
        return

    # Convert Metadata input from string to JSON
    try:
        mdata = json.loads(json_input)
    except json.JSONDecodeError:
        gr.Warning("Input is not valid JSON")
        return

    # If new file name is omitted, replace old file
    if len(new_name) == 0:
        new_name = file_name[0].replace(".safetensors", "_md.safetensors")
    else:
        # Enforce ending in .safetensors
        if not new_name.endswith(".safetensors"):
            new_name = new_name + ".safetensors"

    # Build paths from model path and file name
    file_path = str(os.path.join(model_path, file_name[0]))
    new_path = str(os.path.join(model_path, new_name))

    # check if file already exists
    if os.path.isfile(new_path):
        gr.Warning("Name already existing!")
        return

    log("Saving Checkpoint...")

    # save extracted checkpoints into new file, along with metadata
    write_metadata(file_path, mdata, new_path)

    log("Saved Checkpoint!")


def log(message):
    gr.Info(message)
    print("\n" + message)
