from src.tools.advanced_operations import AdvancedOperations
import torch

class AdvancedToolAgent:
    def __init__(self, config_detection, config_segmentation, config_inpainting):
        self.operations = AdvancedOperations(config_detection, config_segmentation, config_inpainting)
        self.tools = {
            'delete_object': {
                'name': "delete_object",
                'func': self.operations.delete_object,
                'type': 'advanced',
                'description': 'Remove objects using Grounding DINO + SAM + BRIA API',
                'parameters': {
                    'text_prompt': {
                        'type': 'str', 'required': True,
                        'note': 'Prompt has to be lowercase and add a period after the prompt',
                    }
                },
            }
        }

    def get_tools(self):
        return self.tools

    def has_tool(self, tool_name):
        return tool_name in self.tools

    def execute(self, tool_name, image, **params):
        if not self.has_tool(tool_name):
            raise ValueError(f"Advanced tool {tool_name} not supported")
        return self.tools[tool_name]['func'](image, **params)