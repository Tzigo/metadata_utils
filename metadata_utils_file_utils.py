import modules.sd_models as models
import os

import modules.shared

lora_list = []
lora_path = getattr(modules.shared.cmd_opts, "lora_path", os.path.join(models.paths.models_path, "Lora"))


def list_loras():
    global lora_list
    if not os.path.isdir(lora_path):
        print("failed")
        return

    def list_recursive(path: str) -> list[str]:
        out = []
        global_path = os.path.join(lora_path, path)
        for item in os.listdir(global_path):
            if os.path.isfile(os.path.join(global_path, item)):
                out.append(os.path.join(path, item))
            elif os.path.isdir(os.path.join(global_path, item)):
                out.extend(list_recursive(os.path.join(path, item)))

        return out

    lora_list = list_recursive("")


def get_lora(lora):
    if not os.path.isfile(os.path.join(lora_path, lora)):
        return None
    return os.path.join(lora_path, lora)


class LoraFileImitat:
    def __init__(self, lora):
        self.is_safetensors = lora.endswith(".safetensors")
        self.filename = os.path.join(lora_path, lora)


def get_lora_on_button(lora):
    if not os.path.isfile(os.path.join(lora_path, lora)):
        return None
    return LoraFileImitat(lora)


def lora_tiles():
    global lora_list
    if len(lora_list) == 0:
        list_loras()
    return lora_list


list_loras()
