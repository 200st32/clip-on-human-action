
import argparse
import json
import os

import cv2
import torch
from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.transforms import Resize, ConvertImageDtype, Normalize

import myutils
import Mymodel
from transformers import AutoImageProcessor

def preprocess_image(image_path: str, device: torch.device, m_type) -> torch.Tensor:
    image = Image.open(image_path)
   
    if m_type == 'siglip':
        processor = AutoImageProcessor.from_pretrained("google/siglip-base-patch16-224")
        size = processor.size["height"]
        std = processor.image_std
        mean = processor.image_mean
        data_transforms = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
 
        data_transforms = transforms.Compose([
            transforms.Resize(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

    tensor = data_transforms(image)
    tensor = tensor.unsqueeze(0)
    # Transfer tensor channel image format data to CUDA device
    tensor = tensor.to(device=device, memory_format=torch.channels_last, non_blocking=True)

    return tensor


def main(args):
    # Get the label name corresponding to the drawing
    class_label_map = [
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

    # Check if cuda is available
    use_cuda = torch.cuda.is_available()

    # Set proper device based on cuda availability 
    device = torch.device("cuda" if use_cuda else "cpu")

    # Initialize the model
    model = Mymodel.load_model(args.type, device) 
    # Load model weights
    model.load_state_dict(torch.load(args.model_weights_path, weights_only=True))

    # Start the verification mode of the model.
    model.eval()

    for class_name in class_label_map:
        class_path = args.image_path + class_name + '/'
        class_total = 0
        class_correct = 0
        for class_img in os.listdir(class_path): 
            input_img = preprocess_image(class_path+class_img, device, args.type)
            # Inference
            with torch.no_grad():
                if args.type=='siglip':
                    outputs = model(input_img)
                    logits = outputs.logits
                    sigmoid = torch.nn.Sigmoid()
                    output = sigmoid(logits.squeeze())
                else:
                    output = model(input_img)
            # Calculate the highest classification probability

            if args.type == 'siglip':
                prediction_class_index = torch.topk(output, k=1).indices.tolist()
            else:
                prediction_class_index = torch.topk(output, k=1).indices.squeeze(0).tolist()
            
            for class_index in prediction_class_index:
                prediction_class_label = class_label_map[class_index]
            class_total += 1
            if prediction_class_label == class_name:
                class_correct += 1
        class_accuracy = (class_correct/class_total)*100
        print(f"{class_name} accuracy: {class_accuracy:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='clip',help='model type')
    parser.add_argument("--model_mean_parameters", type=list, default=[0.485, 0.456, 0.406])
    parser.add_argument("--model_std_parameters", type=list, default=[0.229, 0.224, 0.225])
    parser.add_argument("--model_weights_path", type=str, default="./myoutput/20_clip_best_model.pth")
    parser.add_argument("--image_path", type=str, default="/home/cap6411.student1/CVsystem/assignment/hw5/human-action-recognition-dataset/Structured/test/")
    parser.add_argument("--image_size", type=int, default=224)
    args = parser.parse_args()
    main(args)

