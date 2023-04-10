import torch
from PIL import Image
# setup device to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load sample image
raw_image = Image.open("Emotion-droplets-Tintin-et-les-Picaros-panel-5022.png").convert("RGB")

from lavis.models import load_model_and_preprocess

model_zoo = {
        "blip_caption":                   ["base_coco", "large_coco"],
#        "blip_classification":            ["base"],
        "blip_feature_extractor":         ["base"],
        "blip_image_text_matching":       ["base", "large"],
        "blip_nlvr":                      ["nlvr"],
        "blip_pretrain":                  ["base"],
        "blip_retrieval":                 ["coco", "flickr"],
        "blip_vqa":                       ["vqav2", "okvqa", "aokvqa"],
        "blip2_opt":                      ["pretrain_opt2.7b", "pretrain_opt6.7b", "caption_coco_opt2.7b", "caption_coco_opt6.7b"],
        "blip2_t5":                       ["pretrain_flant5xl", "pretrain_flant5xl_vitL", "pretrain_flant5xxl", "caption_coco_flant5xl"],
        "blip2_feature_extractor":        ["pretrain", "pretrain_vitL", "coco]"],
        "blip2":                          ["pretrain", "pretrain_vitL", "coco"],
        "blip2_image_text_matching":      ["pretrain", "pretrain_vitL", "coco"],
}


def ImageCaption():
    for name, models in model_zoo.items():
        for model_type in models:
          print("model=" + name + ", type=" + model_type)
          model, vis_processors, _ = load_model_and_preprocess(name=name, model_type=model_type, is_eval=True, device=device)

          # vis_processors stores image transforms for "train" and "eval" (validation / testing / inference)
          image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

          # generate caption
          print(model.generate({"image": image}))

ImageCaption()

