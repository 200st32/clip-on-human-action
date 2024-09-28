import torch
import clip 
import torch.nn as nn
import timm
from transformers import AutoImageProcessor, AutoModelForImageClassification
from A_CLIP import models

def get_att_mask(attention, ratio=0.5):
    bs = attention.shape[0]  
    masks = torch.ones((bs,49), dtype=torch.bool, device=attention.device)
    attention = attention.reshape((-1, 14, 14))
    attention = torch.nn.functional.interpolate(attention.unsqueeze(1), (7, 7), mode='bilinear').squeeze()
    attention = attention.reshape(bs,-1)
    N = int(attention.shape[1] * ratio)

    reservation = torch.argsort(attention, descending=True)
    reservation = reservation[:,:N+1]
    masks = masks.scatter_(1, reservation, False)
 
    full_mask = torch.zeros((bs, 14, 14), dtype=torch.bool, device=attention.device)
    full_mask[:, 0::2, 0::2] = masks.reshape(bs, 7, 7)
    full_mask[:, 0::2, 1::2] = masks.reshape(bs, 7, 7)
    full_mask[:, 1::2, 0::2] = masks.reshape(bs, 7, 7)
    full_mask[:, 1::2, 1::2] = masks.reshape(bs, 7, 7)
    full_mask = full_mask.reshape(bs, -1)

    return full_mask

class MyACLIP(nn.Module):
    def __init__(self, vision_model, vision_model_ema, num_classes):
        super(MyACLIP, self).__init__()
        self.vision_model = vision_model
        self.vision_model_ema = vision_model_ema
        self.classifier = nn.Linear(768, num_classes)

    def encode_image(self, image, mask=None, ret=False, ema=False):
        if ema == False:
            x, attn, _ = self.visual_model(image, mask=mask, need_attn=False)
            tokens = x
            x = x[:, 0] @ self.image_projection
        else:
            x, attn, _ = self.visual_model_ema(image, mask=mask, need_attn=True)
            tokens = x
            x = x[:, 0] @ self.image_projection_e

        if ret:
            return x, attn, tokens
        return x    

    def get_mask(self, mask, positions, e_positions):
        # top, left, width, height = pos

        mask = mask.reshape((-1, 14, 14))
        cmask = []

        for i in range(mask.shape[0]):
            m = mask[i]
            m = m.unsqueeze(0)
            m = m.unsqueeze(0)
            o_pos = positions[i]
            e_pos = e_positions[i]
            m = torch.nn.functional.interpolate(m, (e_pos[2], e_pos[3]), mode='bilinear')

            top = o_pos[0] - e_pos[0]
            left = o_pos[1] - e_pos[1]
            m = F_t.crop(m, top, left, o_pos[2], o_pos[3])
            m = torch.nn.functional.interpolate(m, (14, 14), mode='bilinear')
            cmask.append(m)

        cmask = torch.stack(cmask).squeeze()
        cmask = cmask.reshape(mask.shape[0], -1)
        return cmask
    

    def forward(self, im1, im2, pos):

        with torch.no_grad():
            attn = self.vision_model_ema(im2, need_attn=True)
        attention_map = attn
        attention_map_1 = self.get_mask(attention_map,pos[:,0],pos[:,2])
        mask_1 = get_att_mask(attention_map_1)
        attention_map_2 = self.get_mask(attention_map,pos[:,1],pos[:,2])
        mask_2 = get_att_mask(attention_map_2)
        mask = torch.cat([mask_1,mask_2],dim=0)
        with torch.no_grad():
            image_embed, _, tokens = self.encode_image(im1, mask=mask, ret=True)        

        return self.classifier(image_embed)

class CLIP_Classify(nn.Module):
    def __init__(self, model, num_classes):
        super(CLIP_Classify, self).__init__()
        self.model = model
        self.classifier = nn.Linear(model.visual.output_dim, num_classes)
    
    def forward(self, x):
        with torch.no_grad():
            features = self.model.encode_image(x).float()  # Convert to float32
        return self.classifier(features)

class MyACLIP_Classify(nn.Module):
    def __init__(self, model, num_classes):
        super(MyACLIP_Classify, self).__init__()
        self.model = model
        self.mask = nn.Sequential(
                    nn.Linear(512, 512),
                    nn.Sigmoid()
                )
        self.classifier = nn.Linear(512, num_classes)
    def forward(self, x):
        with torch.no_grad():
            features = self.model.encode_image(x).float()
        mymask = self.mask(features)
        x = mymask * features
        return self.classifier(x)

class MyACLIP_Classify_rand(nn.Module):
    def __init__(self, model, num_classes, device):
        super(MyACLIP_Classify_rand, self).__init__()
        self.model = model
       
        self.mask = (torch.rand(512) < 0.5).int().to(device)
         
        self.classifier = nn.Linear(512, num_classes)
    def forward(self, x):
        with torch.no_grad():
            features = self.model.encode_image(x).float()
        x = self.mask * features
        return self.classifier(x)

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
        #mymodel = timm.models.create_model('vit_base_patch16_224', num_classes=class_num)
        vision_model = timm.create_model('mask_vit_base_patch16_224', num_classes=class_num, mask_ratio=0.2)
        vision_model.load_state_dict(state_dict, strict=False)
        vision_model_ema = timm.create_model('mask_vit_base_patch16_224', num_classes=class_num, mask_ratio=0)
        vision_model_ema.load_state_dict(state_dict, strict=False)
        '''
        # freeze all layers but the last fc
        for name, param in mymodel.named_parameters():
            if name not in ['%s.weight' % linear_keyword, '%s.bias' % linear_keyword]:
                param.requires_grad = False
        # init the fc layer
        getattr(mymodel, linear_keyword).weight.data.normal_(mean=0.0, std=0.01)
        getattr(mymodel, linear_keyword).bias.data.zero_()    
        '''
        mymodel = MyACLIP(vision_model, vision_model_ema, 15)
    elif model_type =='aclip':
        #mymodel = getattr(models, 'ACLIP_VITB16')(mask_ratio=0)
        model, preprocess = clip.load("ViT-B/16", jit=False)
        mymodel = MyACLIP_Classify(model, class_num)

    elif model_type =='aclip_rand':
        model, preprocess = clip.load("ViT-B/16", jit=False)
        mymodel = MyACLIP_Classify_rand(model, class_num, device)


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
    model = load_model("test_aclip", device)


