import google.generativeai as genai
import json
import inspect
from typing import Dict, Any

class Planner:
    def __init__(self, config, tools):
        genai.configure(api_key=config['api']['gemini_key'])
        self.generated_steps = []
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        self.tools = tools
        self.verbose = 0  # Default to silent mode
        self.log_buffer = []
        self.config = config

    def generate_plan(self, goal_description):
        self.generated_steps = []
        self.log_buffer = []
        # tool_list = "\n".join([f"- {name}: {tool['description']}; " for name, tool in self.tools.items()])

        prompt = f"""
Analyze the user's goal and generate processing steps. Follow these RULES:

1. For advanced tools, PARAMETERS MUST BE EXPLICITLY EXTRACTED from the request
2. Required parameters must ALWAYS be included
3. If parameters are missing for advanced tools, SKIP THE STEP

The response should be a JSON object with the following structure:
{{
    "steps": [
        {{
            "tool": "tool_key_name",
            "parameters": {{}},
            "log_message": "Description of the step",
            "metadata": {{
                "tool_type": "basic/advanced",
                "validation_status": "approved/rejected"
            }}
        }}
    ],
    "verbose": 1  # Set to 1 to enable verbose logging, 0
            "validation_status": "approved"
            "validation_status": "rejected"
            to disable
}}

AVAILABLE TOOLS: {self.tools} and please if the user give extra parameters that are not mentioned in the tool description, please ignore them and do not include them in the parameters.
There's are notes in some tools that you should follow, like for delete_object tool, the prompt has to be lowercase and add a colon after the prompt.

EXAMPLE 1:
Request: "Remove the laptop and resize to 800x600, detect edges by canny method with low_threshold=0.2, high_threshold=0.8"
Response:
{{
  "steps": [
    {{
      "tool": "delete_object",
      "parameters": {{"object_name": "a laptop."}},
      "log_message": "Removing laptop"
    }},
    {{
      "tool": "resize_image",
      "parameters": {{"size": [800, 600]}}
    }},
    {{
      "tool": "detect_edges",
      "parameters": {{
        "method": "canny",
        "low_threshold": 0.2,
        "high_threshold": 0.8
      }}
    }}
  ]
}}

USER REQUEST: {goal_description}
"""
        response = self.model.generate_content(prompt)
        response_text = response.text.strip()
        print(response_text)

        # Extract JSON from response
        if response_text.startswith("```json"):
            response_text = response_text[7:-3].strip()

        plan = json.loads(response_text)
        
        # Set verbose mode based on LLM decision
        self.verbose = plan.get("verbose", 0)
        self._log(f"Verbose mode {'activated' if self.verbose else 'disabled'} by LLM")
        
        return self._validate_steps(plan.get("steps", []))

    def _validate_steps(self, steps):
        valid_steps = []
        self._log(f"Available tools:")
        for tool_name, tool_config in self.tools.items():
            self._log(f"- {tool_name}: {tool_config['description']}, {tool_config['parameters']}")
        
        # Fix 1: Use enumerate to track step index
        for step_idx, step in enumerate(steps, start=1):
            tool_name = step.get("tool")
            params = step.get("parameters", {})
            
            # Fix 2: Handle missing log_message safely
            log_msg = step.get("log_message", f"Step {step_idx}")
            metadata = step.get("metadata", {})
            
            if tool_name not in self.tools:
                self._log(f"Skipping unknown tool: {tool_name}", "warning")
                continue
                
            tool_config = self.tools[tool_name]
            required_params = [
                pname for pname, pconfig in tool_config['parameters'].items() 
                if pconfig['required']
            ]
            
            # Check required parameters
            missing_params = [p for p in required_params if p not in params]
            if missing_params:
                self._log(f"Skipping {tool_name} - Missing required params: {missing_params}", "warning")
                continue

            # Fix 3: Proper parameter validation with signature inspection
            try:
                sig = inspect.signature(tool_config['func'])
                valid_params = {}
                
                for param in sig.parameters.values():
                    if param.name == 'image':
                        continue
                    
                    # Handle parameter conversion
                    if param.name in params:
                        param_value = params[param.name]
                        if param.name == 'size' and isinstance(param_value, list):
                            valid_params[param.name] = tuple(param_value)
                        else:
                            valid_params[param.name] = param_value
                    elif param.default != inspect.Parameter.empty:
                        valid_params[param.name] = param.default

                # Fix 4: Validate parameter types
                # for param_name, param_value in valid_params.items():
                #     expected_type = tool_config['parameters'][param_name].get('type')
                #     if expected_type and not isinstance(param_value, eval(expected_type)):
                #         raise TypeError(f"Invalid type for {param_name}. Expected {expected_type}, got {type(param_value)}")

                valid_steps.append({
                    "tool": tool_name,
                    "parameters": valid_params,
                    "log_message": log_msg,
                    "metadata": {
                        **metadata,
                        "tool_type": tool_config.get('type', 'basic'),
                        "validation_status": "approved"
                    }
                })

                if self.verbose:
                    # Fix 5: Use correct step index in logging
                    self._log(f"Validated step {step_idx}: {log_msg}")
                    self._log(f"Parameters: {valid_params}")
                    self._log(f"Metadata: {metadata}")

            except Exception as e:
                self._log(f"Validation failed for step {step_idx}: {str(e)}", "error")
                continue

        self.generated_steps = valid_steps
        return self.generated_steps

    def _log(self, message, level='info'):
        """Controlled logging based on verbose mode"""
        if self.verbose or level in ('warning', 'error'):
            log_entry = f"[{level.upper()}] {message}"
            self.log_buffer.append(log_entry)
            print(log_entry)