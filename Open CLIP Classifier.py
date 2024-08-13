import torch
from PIL import Image
import open_clip
import glob
import os
import json
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_score, f1_score, accuracy_score
import matplotlib.pyplot as plt

json_path = 'Json/Test Light Box [NT,DS,DA].json'

input_data = []

def calculate_recall(y_true, y_pred, class_label):
    # Count true positives, false negatives, and false positives
    TP = sum((true_label == class_label) and (pred_label == class_label) for true_label, pred_label in zip(y_true, y_pred))
    FN = sum((true_label == class_label) and (pred_label != class_label) for true_label, pred_label in zip(y_true, y_pred))
    
    # Calculate recall
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    
    return recall


def argmax(iterable):

    return max(enumerate(iterable), key=lambda x: x[1])[0]

with open(json_path, 'r') as f:

	for line in f:

		obj = json.loads(line)

		input_data.append(obj)

model_path = {
    
    
    ""



}

for models in model_path:

    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained=models)

    tokenizer = open_clip.get_tokenizer('ViT-B-32')

    # candidate_captions = ["A photo of a germinated palm oil seed with sprout of even length,taken under normal room lighting with brighter background, distinguished by the display of a healthy beige color on both structures, indicating a good seed", 
    #                     "A photo of a germinated palm oil seed with sprout of uneven length,taken under normal room lighting with brighter background, distinguished by the presence of brown patches on both structures, indicating a bad seed",
    #                     "A photo of a germinated palm oil seed with sprout of even length,taken in a light box with darker background, distinguished by the display of a healthy beige color on both structures, indicating a good seed",
    #                     "A photo of a germinated palm oil seed with sprout of uneven length,taken in a light box with darker background, distinguished by the presence of brown patches on both structures, indicating a bad seed"]


    candidate_captions = [


        "A photo of a germinated palm oil seed with shoot of even length, distinguished by the display of a creamy beige color on the shoot, indicating a good seed",
        "A photo of a germinated palm oil seed with shoot of uneven length, distinguished by the presence of brown patches on the shoot, indicating a bad seed"

    ]

    text = tokenizer(candidate_captions)

    # Choose computation device
    device = "cuda" if torch.cuda.is_available() else "cpu" 


    list_image_path = []

    list_txt = []

    for item in input_data:

        img_path = item['filename']

        caption = item['captions']

        list_image_path.append(img_path)

        list_txt.append(caption)

    size = len(list_txt)

    counter = 0

    seed_true = []

    seed_pred = []

    progress_bar = tqdm(total=size, desc="Processing")

    while (counter < size):

        image = preprocess(Image.open(list_image_path[counter])).unsqueeze(0)

        with torch.no_grad(), torch.cuda.amp.autocast():

            image_features = model.encode_image(image)

            text_features = model.encode_text(text)

            image_features /= image_features.norm(dim=-1, keepdim=True)

            text_features /= text_features.norm(dim=-1, keepdim=True)

            text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

            top_probs, top_labels = text_probs.cpu().topk(1, dim=-1)

            pred_caption = candidate_captions[top_labels] 

            seed_pred.append(pred_caption)

            seed_true.append(list_txt[counter])

        progress_bar.update(1)

        counter+=1\

    # label_mapping = {"A photo of a germinated palm oil seed with sprout of even length,taken under normal room lighting with brighter background, distinguished by the display of a healthy beige color on both structures, indicating a good seed": 0,
    #                  "A photo of a germinated palm oil seed with sprout of uneven length,taken under normal room lighting with brighter background, distinguished by the presence of brown patches on both structures, indicating a bad seed": 1,
    #                  "A photo of a germinated palm oil seed with sprout of even length,taken in a light box with darker background, distinguished by the display of a healthy beige color on both structures, indicating a good seed": 2,
    #                  "A photo of a germinated palm oil seed with sprout of uneven length,taken in a light box with darker background, distinguished by the presence of brown patches on both structures, indicating a bad seed": 3}

    label_mapping = {

        "A photo of a germinated palm oil seed with shoot of uneven length, distinguished by the presence of brown patches on the shoot, indicating a bad seed":0,
        "A photo of a germinated palm oil seed with shoot of even length, distinguished by the display of a creamy beige color on the shoot, indicating a good seed":1,


    }



    seed_true_numeric = [label_mapping[label] for label in seed_true]

    seed_pred_numeric = [label_mapping[label] for label in seed_pred]

    accuracy = accuracy_score(seed_true_numeric, seed_pred_numeric)

    print(models)

    print(f"Accuracy: {accuracy}")

