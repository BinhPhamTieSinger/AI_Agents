import cv2
import torch
import numpy as np
from PIL import Image
from transformers import pipeline
from diffusers import StableDiffusionXLInpaintPipeline, StableDiffusionInstructPix2PixPipeline
from segment_anything import SamPredictor, sam_model_registry
import tqdm

class AdvancedOperations:
    def __init__(self, config_segmentation, config_inpainting):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Initialize progress bar
        pbar = tqdm.tqdm(total=3, desc="Initializing models")

        # Initialize detection model
        pbar.set_description("Loading OWLv2 detector")
        self.detector = pipeline(
            model=config_segmentation['owlv2']['model'],
            task="zero-shot-object-detection",
            device=self.device
        )
        pbar.update(1)

        # Initialize SAM model
        pbar.set_description("Loading SAM predictor")
        self.sam = sam_model_registry[config_segmentation['sam']['model_type']](
            checkpoint=config_segmentation['sam']['checkpoint_path']
        ).to(self.device)
        self.predictor = SamPredictor(self.sam)
        pbar.update(1)
        self.mask_params = config_segmentation['mask_processing']

        # Initialize inpainting model
        pbar.set_description("Loading inpainting model")
        # self.inpainting_pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        #     config_inpainting['sdxl']['model'],
        #     torch_dtype=getattr(torch, config_inpainting['sdxl']['torch_dtype']),
        #     variant=config_inpainting['sdxl']['variant'],
        #     use_safetensors=config_inpainting['sdxl']['use_safetensors']
        # ).to(self.device)
        # pbar.update(1)

        # pbar.close()
        
        # self.inpainting_params = config_inpainting['inpainting_params']

        self.replacement_params = config_inpainting.get('replacement_params', {
            'blend_strength': 0.9,
            'context_preservation': 0.85
        })

        self.pix2pix_pipe = None
        if "instruct_pix2pix" in config_inpainting:
            self._load_pix2pix(config_inpainting)

    def _load_pix2pix(self, config):
        """Load Instruct-Pix2Pix model separately"""
        
        self.pix2pix_pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            config['instruct_pix2pix']['model'],
            torch_dtype=getattr(torch, config['instruct_pix2pix']['torch_dtype']),
        ).to(self.device)
        self.pix2pix_params = config['instruct_pix2pix']['parameters']

    def _get_pipeline(self, operation_type):
        """Route operations to appropriate pipeline"""
        if operation_type == "edit" and self.pix2pix_pipe:
            return self.pix2pix_pipe, self.pix2pix_params
        return self.inpainting_pipe, self.inpainting_params
    
    @torch.inference_mode()
    def edit_image(self, image, instruction):
        """General-purpose image editing using Instruct-Pix2Pix"""
        if not self.pix2pix_pipe:
            raise ValueError("Instruct-Pix2Pix model not loaded")
            
        return self.pix2pix_pipe(
            prompt=instruction,
            negative_prompt=self.pix2pix_params['negative_prompt'],
            image=image,
            **self.pix2pix_params
        ).images[0]

    def _process_mask(self, mask_array):
        """Enhanced mask processing with morphological operations"""
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (2*self.mask_params['expand_pixels']+1, 
             2*self.mask_params['expand_pixels']+1)
        )
        
        dilated = cv2.dilate(mask_array, kernel, iterations=2)
        blurred = cv2.GaussianBlur(dilated, 
                                 (self.mask_params['blur_radius'], 
                                  self.mask_params['blur_radius']), 0)
        _, processed_mask = cv2.threshold(blurred, 
                                        self.mask_params['mask_threshold'], 
                                        255, 
                                        cv2.THRESH_BINARY)
        return processed_mask

    @torch.inference_mode()
    def delete_object(self, image, object_name):
        """Remove specified objects from the image"""
        image_array = np.array(image.convert("RGB"))
        
        detections = self.detector(
            image, 
            candidate_labels=[object_name], 
            threshold=0.15
        )
        
        if not detections:
            return image
        
        boxes = [
            [det["box"]["xmin"], det["box"]["ymin"], 
             det["box"]["xmax"], det["box"]["ymax"]]
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
        
        combined_mask = torch.any(masks, dim=0).cpu().numpy()[0]
        processed_mask = self._process_mask((combined_mask * 255).astype(np.uint8))
        mask_image = Image.fromarray(processed_mask)
        
        result = self.inpainting_pipe(
            prompt=f"Remove the {object_name}, fill the area with a clean and match with the surrounding background, seamless integration",
            negative_prompt=self.inpainting_params['negative_prompt'],
            image=image,
            mask_image=mask_image,
            num_inference_steps=self.inpainting_params['num_inference_steps'],
            guidance_scale=self.inpainting_params['guidance_scale'],
            strength=self.inpainting_params['strength'],
            denoising_start=self.inpainting_params['denoising_start'],
            output_type="pil"
        ).images[0]
        
        return result
    
    @torch.inference_mode()
    def add_object(self, image, object_description, position_hint=None):
        """Add new objects to the image"""
        # Create mask based on position hint or full image
        if position_hint:
            mask = self._create_mask_from_hint(image, position_hint)
        else:
            mask = self._create_full_image_mask(image)
        
        # Generate with different prompt strategy
        return self.inpainting_pipe(
            prompt=f"High quality photo of {object_description}, perfectly integrated with surroundings",
            image=image,
            mask_image=mask,
            **self.inpainting_params
        ).images[0]

    @torch.inference_mode()
    def replace_object(self, image, old_object_name, new_object_description):
        """Replace existing objects with new ones"""
        # First delete old object
        intermediate_image = self.delete_object(image, old_object_name)
        
        # Then add new object in the same location
        return self.add_object(
            intermediate_image,
            new_object_description,
            position_hint=old_object_name
        )

    def _create_mask_from_hint(self, image, hint):
        """Create mask based on text hint using existing detection"""
        detections = self.detector(
            image,
            candidate_labels=[hint],
            threshold=0.1
        )
        
        # Similar mask creation as delete_object
        boxes = [[det["box"]["xmin"], det["box"]["ymin"], 
                det["box"]["xmax"], det["box"]["ymax"]] for det in detections]
        
        self.predictor.set_image(np.array(image))
        transformed_boxes = self.predictor.transform.apply_boxes_torch(
            torch.tensor(boxes, device=self.device),
            image.size[::-1]  # (H, W)
        )
        
        masks, _, _ = self.predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False
        )
        
        return self._process_combined_mask(masks)

    def _create_full_image_mask(self, image):
        """Create full image mask for global generation"""
        w, h = image.size
        mask = Image.new('L', (w, h), 255)
        return mask

    def _process_combined_mask(self, masks):
        """Reusable mask processing logic"""
        combined_mask = torch.any(masks, dim=0).cpu().numpy()[0]
        processed_mask = self._process_mask((combined_mask * 255).astype(np.uint8))
        return Image.fromarray(processed_mask)