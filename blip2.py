import os
from PIL import Image
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch
import html5lib

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")

# there are several ways to load the model:
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b")
prompt = ""

input_file = 'Gravrøverne/39619_div-12.htm'

with open(input_file, encoding='Windows-1252') as fd:
    doc = html5lib.parse(fd.read(), namespaceHTMLElements=False)

for el in doc.findall(".//div[@class=\"page\"]"):
    imgEl = el.find("./img")
    if imgEl == None:
        continue

    image_file = os.path.join(os.path.dirname(input_file), imgEl.attrib["src"])
    print(image_file)
    image = Image.open(image_file).convert('RGB')
    for areaEl in el.findall("./div[@class=\"area\"]"):
        id = areaEl.attrib["id"]
        top = 0
        left = 0
        width = 0
        height = 0
        for part in areaEl.attrib["style"].split(";"):
            if part == "":
                continue

            (key, valueStr) = part.split(":")
            if (valueStr == ""):
                continue

            valueStr = valueStr.replace("px", "")
            value = int(valueStr)
            # print(key, value)

            if key == "width":
                width = value
            elif key == "height":
                height = value
            elif key == "left":
                left = value
            elif key == "top":
                top = value

        frame = image.crop((left, top, left + width, top + height))
        inputs = processor(frame, text=prompt, return_tensors="pt").to(device)

        generated_ids = model.generate(**inputs, max_new_tokens=20)
        generated_text = processor.batch_decode(
            generated_ids, skip_special_tokens=True)[0].strip()
        print(f"id:{id}, description:{generated_text}")

exit(0)

### ChatGPT prompt:
# I need you to clean up descriptions of comic book panels, I will provide the text inside {} and you will reply with Description: {}

# The description is meant to be short and of the visuals only. It is always about a single panel. Remove mentions of speech bubbles, panels, strips, or other comic book terminology. 