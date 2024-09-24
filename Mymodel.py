import torch
import clip 
import torch.nn as nn
import timm
from transformers import AutoImageProcessor, AutoModelForImageClassification
from A_CLIP import models

class CLIP_Classify(nn.Module):
    def __init__(self, model, num_classes):
        super(CLIP_Classify, self).__init__()
        self.model = model
        self.classifier = nn.Linear(model.visual.output_dim, num_classes)
    
    def forward(self, x):
        with torch.no_grad():
            features = self.model.encode_image(x).float()  # Convert to float32
        return self.classifier(features)

def load_model(model_type, device, class_num=15):
    if model_type == 'clip':
        model, preprocess = clip.load("ViT-B/16", jit=False)
        mymodel = CLIP_Classify(model, class_num)
        
    elif model_type == 'test_aclip':
        linear_keyword = 'head'
        checkpoint = torch.load('./pretrained/ViT_B_16_vision_model.pt', weights_only=True)
        visual_keyword = 'module.visual.'
        # rename CLIP pre-trained keys
        state_dict = checkpoint
        for k in list(state_dict.keys()):
            # retain only base_encoder up to before the embedding layer
            if k.startswith(visual_keyword) and not k.startswith(visual_keyword + linear_keyword):
                # remove prefix
                state_dict[k[len(visual_keyword):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]
        mymodel = timm.models.create_model('vit_base_patch16_224', num_classes=class_num)
        mymodel.load_state_dict(state_dict, strict=False)
        # freeze all layers but the last fc
        for name, param in mymodel.named_parameters():
            if name not in ['%s.weight' % linear_keyword, '%s.bias' % linear_keyword]:
                param.requires_grad = False
        # init the fc layer
        getattr(mymodel, linear_keyword).weight.data.normal_(mean=0.0, std=0.01)
        getattr(mymodel, linear_keyword).bias.data.zero_()    

    elif model_type =='aclip':
        mymodel = getattr(models, 'ACLIP_VITB16')(mask_ratio=0)


    elif model_type == 'siglip':
        labels = [
            "calling",
            "clapping",
            "cycling",
            "dancing",
            "drinking",
            "eating",
            "fighting",
            "hugging",
            "laughing",
            "listening_to_music",
            "running",
            "sitting",
            "sleeping",
            "texting",
            "using_laptop"
        ]
        id2label = {id: label for id, label in enumerate(labels)}
        model_id = "google/siglip-base-patch16-224"
        mymodel = AutoModelForImageClassification.from_pretrained(model_id, problem_type="multi_label_classification", id2label=id2label)

    mymodel.to(device)
    print("model load successfully")
    return mymodel

if __name__ == '__main__':
   
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("check model")
    model = load_model("aclip", device)


