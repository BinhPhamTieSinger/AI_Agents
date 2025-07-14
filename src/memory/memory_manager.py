class EditingHistory:
    def __init__(self, max_steps=10):
        self.history = []
        
    def add_operation(self, operation, params):
        self.history.append({
            'operation': operation,
            'params': params
        })
    
    def get_history(self):
        return self.history.copy()