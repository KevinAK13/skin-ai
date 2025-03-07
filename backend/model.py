import torch
import torch.nn as nn
import torchvision.models as models

class SkinCancerModel(nn.Module):
    def __init__(self, num_classes=2):
        super(SkinCancerModel, self).__init__()
        
        # **CNN for images (EfficientNet)**
        self.cnn = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.cnn.classifier = nn.Identity()  # Remove the final layer

        # **MLP for tabular data (age and sex)**
        self.mlp = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU()
        )

        # **Automatically calculate EfficientNet output size**
        dummy_input = torch.randn(1, 3, 224, 224)
        img_output_size = self.cnn(dummy_input).shape[-1]  # Calculate the correct size

        # **Final combined classification layer**
        self.classifier = nn.Sequential(
            nn.Linear(img_output_size + 8, 256),  #  Dynamically adjusted
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, image, metadata):
        img_features = self.cnn(image)  
        meta_features = self.mlp(metadata)
        combined = torch.cat((img_features, meta_features), dim=1)  
        output = self.classifier(combined)
        return output