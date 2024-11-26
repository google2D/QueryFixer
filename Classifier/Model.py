from QueryClassifier import QueryClassifier
import torch
import torch.nn as nn

class Model:
    def __init__(self, model_path):
        self.model = self.load_model(model_path)
        self.freeze_batchnorm()
        self.model.eval()

    # Function to load the classification model
    def load_model(self, model_path):
        model = QueryClassifier(input_size=768)
        model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
        return model

    def freeze_batchnorm(self):
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                m.eval()  # Set to eval mode to stop updating the stats
                # Freeze the running stats if necessary
                m.running_mean = m.running_mean.detach()
                m.running_var = m.running_var.detach()
                m.num_batches_tracked = m.num_batches_tracked.detach()  # Ensure num_batches_tracked is not updated


    def is_well_formed(self, embedded_query_tensor):
        batch_input = embedded_query_tensor.repeat(100, 1)
        with torch.no_grad():
            output = self.model(batch_input)
        return torch.argmax(output, dim=1).data[0]
    