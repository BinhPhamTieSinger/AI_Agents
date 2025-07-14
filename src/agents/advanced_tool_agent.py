from src.tools.advanced_operations import AdvancedOperations

class AdvancedToolAgent:
    def __init__(self, config_segmentation, config_inpainting):
        self.operations = AdvancedOperations(config_segmentation, config_inpainting)
        self.tools = {
            # 'delete_object': {
            #     'name': "delete_object",
            #     'func': self.operations.delete_object,
            #     'type': 'advanced',
            #     'description': 'Remove objects from image',
            #     'parameters': {
            #         'object_name': {'type': 'str', 'required': True}
            #     }
            # },
            # 'add_object': {
            #     'name': "add_object",
            #     'func': self.operations.add_object,
            #     'type': 'advanced',
            #     'description': 'Add new objects to the image',
            #     'parameters': {
            #         'object_description': {'type': 'str', 'required': True},
            #         'position_hint': {'type': 'str', 'required': False}
            #     }
            # },
            # 'replace_object': {
            #     'name': "replace_object",
            #     'func': self.operations.replace_object,
            #     'type': 'advanced',
            #     'description': 'Replace existing objects with new ones',
            #     'parameters': {
            #         'old_object_name': {'type': 'str', 'required': True},
            #         'new_object_description': {'type': 'str', 'required': True}
            #     }
            # },
            'edit_image': {
                'name': "edit_image",
                'func': self.operations.edit_image,
                'type': 'advanced',
                'description': 'Edit image based on instruction',
                'parameters': {
                    'instruction': {'type': 'str', 'required': True}
                }
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