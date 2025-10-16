import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from transformers import AutoModel, AutoTokenizer


class TextEncoder(nn.Module):
    def __init__(self, model_name, device):
        super().__init__()
        self.model, _ = clip.load(model_name, device=device)
        self.device = device
        self.prompt_background = ['{}', 'flawless {}', 'perfect {}', 'healthy {}', '{} without flaw', '{} without illness', '{} without lesion', '{} without tumor']
        self.prompt_foreground = ['diseased {}', 'anomalous {}', '{} with flaw', '{} with illness', '{} with lesion', '{} with tumor']
        self.prompt_state = [self.prompt_background, self.prompt_foreground]
        self.prompt_templates = ['this is one {} in the scene.', 'this is a {} in the scene.', 'this is the {} in the scene.',
                                 'there is a {} in the scene.', 'there is the {} in the scene.',
                                 'a 3D photo of one {}.', 'a 3D photo of a {}.', 'a 3D photo of the {}.',
                                 'a 3D photo of a large {}.', 'a 3D photo of the large {}.', 
                                 'a 3D photo of a small {}.', 'a 3D photo of the small {}.', 
                                 'a 3D photo of a rotated {}.', 'a 3D photo of the rotated {}.', 
                                 'a flipped 3D photo of a {}.', 'a flipped 3D photo of the {}.', 
                                 'a bright 3D photo of a {}.', 'a bright 3D photo of the {}.', 
                                 'a dark 3D photo of a {}.', 'a dark 3D photo of the {}.', 
                                 'a good 3D photo of a {}.', 'a good 3D photo of the {}.',
                                 'a bad 3D photo of a {}.', 'a bad 3D photo of the {}.',  
                                 'a corrupted 3D photo of a {}.', 'a corrupted 3D photo of the {}.',
                                 'a blurry 3D photo of a {}.', 'a blurry 3D photo of the {}.',
                                 'a low resolution 3D photo of a {}.', 'a low resolution 3D photo of the {}.', 
                                 'a close-up 3D photo of a {}.', 'a close-up 3D photo of the {}.', 
                                 'a cropped 3D photo of a {}.', 'a cropped 3D photo of the {}.']

    def forward(self, obj_text):
        text_features = []
        for i in range(len(self.prompt_state)):
            prompted_state = [state.format(obj_text) for state in self.prompt_state[i]]
            prompted_sentence = []
            for s in prompted_state:
                for template in self.prompt_templates:
                    prompted_sentence.append(template.format(s))
            prompted_sentence = clip.tokenize(prompted_sentence).to(self.device)
            class_embeddings = self.model.encode_text(prompted_sentence)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            text_features.append(class_embedding)
        text_features = torch.stack(text_features, dim=1).to(self.device)
        return text_features


