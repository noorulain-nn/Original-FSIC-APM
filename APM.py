# Memory Module
import torch
import torch.nn as nn
import torch.nn.functional as F
class MemoryModule(nn.Module):
    def __init__(self, num_classes, feature_dim):
        super(MemoryModule, self).__init__()
        self.memory = nn.Parameter(torch.randn(num_classes, feature_dim), requires_grad=False)  # Memory matrix
        self.memory_labels = [-1] * num_classes # Initialize all slots as unassigned (-1)
        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.memory, mean=0, std=0.01)
        
    def forward(self, features):
        
        normalized_features = F.normalize(features, p=2, dim=1)  # Shape: [batch_size, feature_dim]
        normalized_memory = F.normalize(self.memory, p=2, dim=1)  # Shape: [num_classes, feature_dim]
    
        # Compute cosine similarity scores between features and memory slots
        # Resulting shape: [batch_size, num_classes]
        
        scores = torch.matmul(normalized_features, normalized_memory.t())
        logits = scores 
        
        predicted_slot = torch.argmax(logits, dim=1)
        predicted_label = [self.memory_labels[slot.item()] for slot in predicted_slot]
        return logits, predicted_slot, predicted_label

    

    # Memory Update Mechanism
    def update_memory(self, features, attention_scores, true_label, predicted_slot):
        
        update_count = 0
        
        # If this is the first time the true label is encountered
        if true_label not in self.memory_labels:

            # If the highest attention slot is available, assign the label to this slot
            if self.memory_labels[predicted_slot] == -1:
                self.memory_labels[predicted_slot] = true_label
                true_slot = self.memory_labels.index(true_label)
                self.memory.data[true_slot] = (
                    F.normalize(features, p=2, dim=-1)
                )
                return 0

            else:
                true_slot = self.memory_labels.index(-1)
                self.memory_labels[true_slot] = true_label
                self.memory.data[true_slot] = F.normalize(features, p=2, dim=-1)
                
                return 1

        true_slot = self.memory_labels.index(true_label)
        #"""
        similarity = F.cosine_similarity(F.normalize(features, p=2, dim=-1), F.normalize(self.memory.data[true_slot], p=2, dim=-1), dim=-1)
        adaptive_lr = 1 - similarity
        self.memory.data[true_slot] = (
            (1 - adaptive_lr) * self.memory.data[true_slot]
            + (adaptive_lr * F.normalize(features, p=2, dim=-1))
        )#"""

        
        return 0       
 
        
# Full Model
class MemoryEnabledCNN(nn.Module):
    def __init__(self, backbone, num_classes, feature_dim):
        super(MemoryEnabledCNN, self).__init__()
        self.backbone = backbone
        self.memory_module = MemoryModule(num_classes, feature_dim)

    def forward(self, x):
        features = self.backbone(x)
        attention_scores, predicted_slot, predicted_label = self.memory_module(features)
        return predicted_label, features, attention_scores, predicted_slot






