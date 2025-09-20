import torch
import torchvision
from PIL import Image
import numpy as np
import torchvision.transforms as T

# Load pretrained Mask R-CNN model (trained on COCO)
def load_model():
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
    model.eval()
    return model

transform = T.Compose([T.ToTensor()])

def remove_background(input_path, output_path, keep_class="person", score_thresh=0.8):
    model = load_model()

    image = Image.open(input_path).convert("RGB")
    img_tensor = transform(image)

    with torch.no_grad():
        predictions = model([img_tensor])

    # COCO class labels
    COCO_CLASSES = [
        "__background__", "person", "bicycle", "car", "motorcycle", "airplane", "bus",
        "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", 
        "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", 
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", 
        "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", 
        "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", 
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", 
        "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", 
        "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", 
        "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", 
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", 
        "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    ]

    masks = predictions[0]["masks"]
    labels = predictions[0]["labels"]
    scores = predictions[0]["scores"]

    # Create a blank mask
    final_mask = np.zeros((image.height, image.width), dtype=np.uint8)

    for i in range(len(masks)):
        if scores[i] > score_thresh:
            label = COCO_CLASSES[labels[i]]
            if label == keep_class:  # keep only person (or other class)
                mask = masks[i, 0].mul(255).byte().cpu().numpy()
                mask = Image.fromarray(mask).resize(image.size)
                mask = np.array(mask) > 128
                final_mask[mask] = 255

    # Apply mask
    image_np = np.array(image)
    result = np.zeros((image_np.shape[0], image_np.shape[1], 4), dtype=np.uint8)
    result[:, :, :3] = image_np
    result[:, :, 3] = final_mask  # alpha channel

    Image.fromarray(result).save(output_path)
