from src.tools.basic_operations import BasicOperations

class BasicToolAgent:
    def __init__(self, config=None):
        self.operations = BasicOperations(config)
        self.tools = {
            'resize_image': {
                'name': "resize",
                'func': self.operations.resize_image,
                'type': 'basic',
                'description': 'Resize image to specified dimensions',
                'parameters': {
                    # 'image': {'type': 'image', 'description': 'Input image', 'required': True},
                    'size': {'type': 'tuple', 'description': 'New size as [width, height]', 'required': False, 'default': [800, 600]},
                }
            },
            'convert_to_grayscale': {
                'name': "grayscale",
                'func': self.operations.convert_to_grayscale,
                'type': 'basic',
                'description': 'Convert image to grayscale',
                'parameters': {
                    # 'image': {'type': 'image', 'description': 'Input image', 'required': True},
                }
            },
            'detect_edges': {
                'name': "edge_detection",
                'func': self.operations.detect_edges,
                'type': 'basic',
                'description': 'Detect edges using Canny/Sobel method',
                'parameters': {
                    # 'image': {'type': 'image', 'description': 'Input image', 'required': True},
                    'method': {
                        'type': 'str',
                        'description': 'Edge detection method (canny/sobel)',
                        'required': False,
                        'default': 'canny'
                    },
                    'low_threshold': {
                        'type': 'int',
                        'description': 'Canny low threshold',
                        'required': False,
                        'default': 50
                    },
                    'high_threshold': {
                        'type': 'int',
                        'description': 'Canny high threshold',
                        'required': False,
                        'default': 200
                    }
                }
            },
            'blur': {
                'name': "blur",
                'func': self.operations.blur,
                'type': 'basic',
                'description': 'Apply Gaussian blur',
                'parameters': {
                    # 'image': {'type': 'image', 'description': 'Input image', 'required': True},
                    'kernel_size': {
                        'type': 'list',
                        'description': 'Kernel size as [width, height]',
                        'required': False,
                        'default': [5, 5]
                    }
                }
            }
        }

    def get_tools(self):
        return self.tools

    def has_tool(self, tool_name):
        return tool_name in self.tools

    def execute(self, tool_name, image, **params):
        if not self.has_tool(tool_name):
            raise ValueError(f"Basic tool {tool_name} not found")
        return self.tools[tool_name]['func'](image, **params)