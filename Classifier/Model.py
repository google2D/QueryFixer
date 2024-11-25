import QueryClassifier
import torch

class Model:
    def __init__(self, model_path):
        self.model = load_model(model_path)

    # Function to load the classification model and set to evaluation mode
    def load_model(self, model_path):
        model = QueryClassifier()
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model

    def is_well_formed(self, embedded_query_tensor):
        return self.model(embedded_query_tensor)