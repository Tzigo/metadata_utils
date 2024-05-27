import gradio as gr
import modules.sd_models as models
from modules.ui import create_refresh_button
from safetensors.torch import save_file, load_file
import os

import re
import json

from modules import script_callbacks

def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as ui_component:
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

        with gr.Row() as row:
            gr.Column(scale=1)
            with gr.Column(scale=1):
                button = gr.Button(value="Add Metadata", variant="primary")
            gr.Column(scale=1)

        button.click(on_button, inputs=[input_file, new_name, json_input, label1])
        return [(ui_component, "Metadata Adder", "extension_template_tab")]


script_callbacks.on_ui_tabs(on_ui_tabs)


def on_button(input_file: str, new_name: str, json_input: str):

    Isforge = False
    getforge = "Forge"
    mainpath = os.getcwd()
    if getforge in mainpath:
            Isforge = True
            webUIbatpath = mainpath + "\\" + "webui-user.bat"

            with open(webUIbatpath, 'r', encoding='utf-8') as file:
                text = file.read()        
            starts_with = '--ckpt-dir "'
            ends_with = '"'
    
            start_index = text.find(starts_with) + len(starts_with)
            end_index = text.find(ends_with, start_index)
            if start_index != -1 and end_index != -1:
                automatic1111ckptfolder = text[start_index:end_index]  

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

    # Build path from global model path and first extracted file name
    file_path = models.model_path + "\\" + file_name[0]

    # If new file name is omitted, replace old file
    if len(new_name) == 0:
        new_name = file_name[0].replace(".safetensors", "_md.safetensors")
    else:
        # Enforce ending in .safetensors
        if not new_name.endswith(".safetensors"):
            new_name = new_name + ".safetensors"

    # Build new path from global model path and new file name
    if Isforge == False:
        new_path = models.model_path + "\\" + new_name
    else:     
        forge_filename = input_file.split(".safetensors")[0] + ".safetensors"       
        filehavefound = False
        
        for dirpath, dirnames, filenames in os.walk(automatic1111ckptfolder):
            if forge_filename in filenames:
                forge_filepath = os.path.join(dirpath, forge_filename)
                filehavefound = True
                break
                
            else:
               filehavefound = False
               return
                
        if filehavefound != True:
            gr.Warning("file not found")
        else:
            gr.Info("Saving Checkpoint...")
            print("\nSaving Checkpoint...")

    # extract tensors from selected checkpoint
            if Isforge:
                loaded_tensors = load_file(forge_filepath) 
            else:   
                loaded_tensors = load_file(file_path)

    # save extracted checkpoints into new file, along with metadata
            if Isforge:
                save_file(loaded_tensors, automatic1111ckptfolder + "\\" + new_name, mdata)
            else:
                save_file(loaded_tensors, new_path, mdata)

            gr.Info("Saved Checkpoint!")
            print("\nSaved Checkpoint!")

        



    
