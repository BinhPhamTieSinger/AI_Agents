class TaskClassifierAgent:
    def __init__(self, planner, basic_agent, advanced_agent):
        self.planner = planner
        self.basic_agent = basic_agent
        self.advanced_agent = advanced_agent

    def process_request(self, image, user_prompt):
        steps = self.planner.generate_plan(user_prompt)
        current_image = image.copy()
        
        for step in steps:
            tool_name = step['tool']
            params = step['parameters']
            
            if self.basic_agent.has_tool(tool_name):
                current_image = self.basic_agent.execute(tool_name, current_image, **params)
            elif self.advanced_agent.has_tool(tool_name):
                current_image = self.advanced_agent.execute(tool_name, current_image, **params)
            else:
                raise ValueError(f"Unknown tool: {tool_name}")
        
        return current_image