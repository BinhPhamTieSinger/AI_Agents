import cv2
import torch
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import base64
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from segment_anything import sam_model_registry, SamPredictor
import tqdm
import subprocess
from pathlib import Path

class AdvancedOperations:
    def __init__(self, config_detection, config_segmentation, config_inpainting):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.det_config = config_detection
        self.seg_config = config_segmentation
        self.api_config = config_inpainting["bria_api"]

        self._load_detector()
        self._load_sam()

        # Initialize detection and segmentation models
        # self.detector = pipeline(
        #     model=config_segmentation['owlv2']['model'],
        #     task="zero-shot-object-detection",
        #     device=self.device
        # )
        
        # self.sam = sam_model_registry[config_segmentation['sam']['model_type']](
        #     checkpoint=config_segmentation['sam']['checkpoint_path']
        # ).to(self.device)
        # self.predictor = SamPredictor(self.sam)
        
        # MI-GAN configuration
        # self.mi_gan_config = config_inpainting['mi_gan']
        # self._prepare_temp_dirs()
        # self._cleaned_up = False
        
    def _load_detector(self):
        """Load Grounding DINO detector"""
        self.detector_processor = AutoProcessor.from_pretrained(
            self.det_config["grounding_dino"]["model"]
        )
        self.detector = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.det_config["grounding_dino"]["model"]
        ).to(self.device)

    def _load_sam(self):
        """Load SAM segmentation model"""
        self.sam = sam_model_registry[self.seg_config["sam"]["model_type"]](
            checkpoint=self.seg_config["sam"]["checkpoint_path"]
        ).to(self.device)
        self.predictor = SamPredictor(self.sam)

    def _prepare_temp_dirs(self):
        """Create temporary directories for MI-GAN processing"""
        self.temp_dirs = {
            'input': Path(self.mi_gan_config['temp_dirs']['input']),
            'mask': Path(self.mi_gan_config['temp_dirs']['mask']),
            'output': Path(self.mi_gan_config['temp_dirs']['output'])
        }
        
        for dir in self.temp_dirs.values():
            dir.mkdir(parents=True, exist_ok=True)

    def _expand_mask(self, mask_array, image_size):
        """Proportional mask expansion"""
        width, height = image_size
        dilation_w = int(width * self.seg_config["mask_processing"]["dilation_proportion"])
        dilation_h = int(height * self.seg_config["mask_processing"]["dilation_proportion"])
        
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2*dilation_w+1, 2*dilation_h+1)
        )
        return cv2.dilate(mask_array, kernel, iterations=1)

    @torch.inference_mode()
    def delete_object(self, image, text_prompt):
        """Enhanced object removal pipeline"""
        # Convert and validate image
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        width, height = image.size
        
        # Detect objects
        inputs = self.detector_processor(
            images=image,
            text=text_prompt,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.detector(**inputs)
            
        results = self.detector_processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids,
            box_threshold=self.det_config["grounding_dino"]["box_threshold"],
            text_threshold=self.det_config["grounding_dino"]["text_threshold"],
            target_sizes=[(height, width)]
        )[0]
        
        boxes = results["boxes"].cpu().numpy()
        if len(boxes) == 0:
            return image
        
        # Expand boxes
        exp_rate = self.det_config["grounding_dino"]["expansion_rate"]
        expanded_boxes = [
            [
                max(0, x1 - width*exp_rate), 
                max(0, y1 - height*exp_rate),
                min(width, x2 + width*exp_rate), 
                min(height, y2 + height*exp_rate)
            ] for x1, y1, x2, y2 in boxes
        ]
        
        # Generate masks
        image_np = np.array(image)
        self.predictor.set_image(image_np)
        boxes_tensor = torch.tensor(expanded_boxes, device=self.device)
        transformed_boxes = self.predictor.transform.apply_boxes_torch(
            boxes_tensor, image_np.shape[:2]
        )
        
        masks, _, _ = self.predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False
        )
        
        # Process mask
        combined_mask = torch.any(masks, dim=0).cpu().numpy()[0]
        mask_array = (combined_mask * 255).astype(np.uint8)
        expanded_mask = self._expand_mask(mask_array, (width, height))
        
        # Call BRIA API
        return self._call_bria_api(image, expanded_mask)
    
    def _call_bria_api(self, image, mask_array):
        """Handle API communication with BRIA"""
        def encode_image(img):
            buffer = BytesIO()
            img.save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        mask = Image.fromarray(mask_array)
        payload = {
            "file": f",{encode_image(image)}",
            "mask_file": f",{encode_image(mask)}",
            "mask_type": self.api_config["mask_type"]
        }
        
        response = requests.post(
            self.api_config["endpoint"],
            json=payload,
            headers={"api_token": self.api_config["api_token"]},
            timeout=self.api_config["timeout"]
        )
        
        response.raise_for_status()
        return Image.open(BytesIO(requests.get(response.json()["result_url"]).content))
    

    def __del__(self):
        if not self._cleaned_up:
            self.cleanup()

    def cleanup(self):
        if self._cleaned_up:
            return
        
        del self.detector
        del self.sam
        del self.predictor
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        for dir in self.temp_dirs.values():
            for file in dir.glob("*"):
                try:
                    file.unlink(missing_ok=True)
                except:
                    pass
        
        self._cleaned_up = True
    
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()