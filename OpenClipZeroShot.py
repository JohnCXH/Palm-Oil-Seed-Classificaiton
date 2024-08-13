import torch
from PIL import Image
import open_clip
import glob
import os



model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32')

tokenizer = open_clip.get_tokenizer('ViT-B-32')

text = tokenizer(["A photo of a good germinated palm oil seed", "A photo of a bad germinated palm oil seed"])

image = preprocess(Image.open("dataset/test/GoodSeed/0.png")).unsqueeze(0)

with torch.no_grad(), torch.cuda.amp.autocast():

    image_features = model.encode_image(image)

    text_features = model.encode_text(text)

    image_features /= image_features.norm(dim=-1, keepdim=True)

    text_features /= text_features.norm(dim=-1, keepdim=True)

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print("Label probs:", text_probs)  # prints: [[1., 0., 0.]]
