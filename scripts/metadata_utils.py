import gradio as gr
import modules.sd_models as models
from modules.ui import create_refresh_button
from metadata_util_lib import write_metadata
from pathlib import Path
import json

from modules import script_callbacks


def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as ui_component:
        with gr.Tab("Adder"):
            with gr.Row():
                input_file = gr.Dropdown(models.checkpoint_tiles(), label="Checkpoint")
                create_refresh_button(input_file, models.list_models,
                                      lambda: {"choices": models.checkpoint_tiles()}, "metadata_utils_refresh_1")

            with gr.Row():
                new_name = gr.Textbox(
                    placeholder='(Optional) Enter new checkpoint name. If omitted, appends "_md" to name',
                    max_lines=1, label="New Name")

            with gr.Row():
                json_input = gr.Code(placeholder='Input JSON content', max_lines=10,
                                     label="Metadata as JSON", language="json")

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
                                      lambda: {"choices": models.checkpoint_tiles()}, "metadata_utils_refresh_2")
            with gr.Row():
                metadata = gr.Code(label="Metadata", language="json")

            input_file.change(
                fn=on_button_load_metadata,
                inputs=[input_file],
                outputs=[metadata]
            )

    return [(ui_component, "Metadata Utils", "metadata_utils_tab")]


script_callbacks.on_ui_tabs(on_ui_tabs)


def on_button_load_metadata(input_file: str):
    if selected_model := models.get_closet_checkpoint_match(input_file):
        if metadata := models.read_metadata_from_safetensors(selected_model.filename):
            return json.dumps(metadata, indent=4, ensure_ascii=False)
        return 'No metadata'
    return 'Model not found'


def on_button(input_file: str, new_name: str, json_input: str):
    selected_model = models.get_closet_checkpoint_match(input_file)
    if not selected_model:
        gr.Warning("Please select a checkpoint")
        return

    if not selected_model.is_safetensors:
        gr.Warning("Selected model is not a .safetensors file")
        return

    selected_model_path = Path(selected_model.filename)

    # If new_name is omitted, replace old file
    new_path = selected_model_path.with_stem(new_name) if new_name else selected_model_path.with_stem(f'{selected_model_path.stem}_md')

    if new_path.exists():
        gr.Warning("Name already existing!")
        return

    # Convert Metadata input from string to JSON
    try:
        mdata = json.loads(json_input)
    except json.JSONDecodeError:
        gr.Warning("Input is not valid JSON")
        return

    log("Saving Checkpoint...")
    # save extracted checkpoints into new file, along with metadata
    write_metadata(selected_model.filename, mdata, str(new_path))
    log("Saved Checkpoint!")


def log(message):
    gr.Info(message)
    print("\n" + message)
