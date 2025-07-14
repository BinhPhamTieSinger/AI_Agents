import cv2
import torch
import numpy as np
from PIL import Image
from transformers import pipeline
from diffusers import StableDiffusionXLInpaintPipeline, StableDiffusionInstructPix2PixPipeline
from segment_anything import SamPredictor, sam_model_registry
import tqdm
import subprocess
from pathlib import Path

class AdvancedOperations:
    def __init__(self, config_segmentation, config_inpainting):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Initialize detection and segmentation models
        self.detector = pipeline(
            model=config_segmentation['owlv2']['model'],
            task="zero-shot-object-detection",
            device=self.device
        )
        
        self.sam = sam_model_registry[config_segmentation['sam']['model_type']](
            checkpoint=config_segmentation['sam']['checkpoint_path']
        ).to(self.device)
        self.predictor = SamPredictor(self.sam)
        
        # MI-GAN configuration
        self.mi_gan_config = config_inpainting['mi_gan']
        self._prepare_temp_dirs()
        self._cleaned_up = False
        
    def _prepare_temp_dirs(self):
        """Create temporary directories for MI-GAN processing"""
        self.temp_dirs = {
            'input': Path(self.mi_gan_config['temp_dirs']['input']),
            'mask': Path(self.mi_gan_config['temp_dirs']['mask']),
            'output': Path(self.mi_gan_config['temp_dirs']['output'])
        }
        
        for dir in self.temp_dirs.values():
            dir.mkdir(parents=True, exist_ok=True)

    def _expand_mask(self, mask_array):
        """Enhanced mask expansion with configurable parameters"""
        kernel_size = 2 * self.mi_gan_config['parameters']['expand_pixels'] + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        return cv2.dilate(
            mask_array, 
            kernel, 
            iterations=self.mi_gan_config['parameters']['dilation_iterations']
        )

    @torch.inference_mode()
    def delete_object(self, image, object_name):
        """Remove objects using MI-GAN with enhanced mask processing"""
        # Convert image to array and detect objects
        image_array = np.array(image.convert("RGB"))
        
        detections = self.detector(
            image, 
            candidate_labels=[object_name], 
            threshold=self.mi_gan_config['parameters']['detection_threshold']
        )
        
        if not detections:
            return image

        # Generate masks with SAM
        boxes = [
            [det["box"]["xmin"], det["box"]["ymin"], det["box"]["xmax"], det["box"]["ymax"]]
            for det in detections
        ]
        
        self.predictor.set_image(image_array)
        transformed_boxes = self.predictor.transform.apply_boxes_torch(
            torch.tensor(boxes, device=self.device),
            image_array.shape[:2]
        )
        
        masks, _, _ = self.predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )
        
        # Process and expand mask
        combined_mask = torch.any(masks, dim=0).cpu().numpy()[0]
        mask_uint8 = (combined_mask * 255).astype(np.uint8)
        expanded_mask = self._expand_mask(mask_uint8)

        # Prepare temporary files
        base_name = hash(image.tobytes())
        input_path = self.temp_dirs['input'] / f"test.png"
        mask_path = self.temp_dirs['mask'] / f"test.png"
        output_path = self.temp_dirs['output'] / f"test.png"

        image.save(input_path)
        Image.fromarray(expanded_mask).save(mask_path)

        # Run MI-GAN
        subprocess.run([
            "python", "-m", "src.scripts.demo",
            "--model-name", "migan-512",
            "--model-path", self.mi_gan_config['model_path'],
            "--images-dir", str(self.temp_dirs['input']),
            "--masks-dir", str(self.temp_dirs['mask']),
            "--output-dir", str(self.temp_dirs['output']),
            "--device", self.device,
            "--invert-mask"
        ], check=True)

        return Image.open(output_path)
    

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