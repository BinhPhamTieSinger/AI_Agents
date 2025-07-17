from src.agents.advanced_tool_agent import AdvancedToolAgent
from src.agents.basic_tool_agent import BasicToolAgent
from src.agents.task_classifier_agent import TaskClassifierAgent
from src.planners.task_planner import Planner
from PIL import Image

class ImageEditor:
    def __init__(self, config, config_detection, config_segmentation, config_inpainting):
        self.config = config

        # Initialize agents
        self.basic_agent = BasicToolAgent()
        self.advanced_agent = AdvancedToolAgent(config_detection, config_segmentation, config_inpainting)
        
        # Collect all available tools
        all_tools = {}
        all_tools.update(self.basic_agent.get_tools())
        all_tools.update(self.advanced_agent.get_tools())
        
        # Initialize planner with tools
        self.planner = Planner(
            config = self.config,
            tools=all_tools
        )
        
        # Initialize classifier
        self.classifier = TaskClassifierAgent(
            self.planner,
            self.basic_agent,
            self.advanced_agent
        )

    def edit_image(self, image_path, prompt):
        image = Image.open(image_path)
        return self.classifier.process_request(image, prompt)
