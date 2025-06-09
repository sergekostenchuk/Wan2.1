import torch
from diffusers import WanPipeline, UniPCMultistepScheduler
from diffusers.utils import export_to_video
import tempfile
import os
import base64

class InferlessPythonModel:
    def initialize(self):
        """Инициализация модели с оптимизацией для A10 24GB"""
        model_id = "Wan-AI/Wan2.1-T2V-1.3B-diffusers"  # Используем 1.3B для стабильности[2]
        
        # Загрузка pipeline
        self.pipe = WanPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,  # Рекомендуемый тип из документации[2]
            variant="fp16"
        )
        
        # Правильный scheduler с flow_shift[2]
        flow_shift = 3.0  # 3.0 для 480P, 5.0 для 720P
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(
            self.pipe.scheduler.config, 
            flow_shift=flow_shift
        )
        
        # Оптимизации памяти для A10[3]
        self.pipe.enable_sequential_cpu_offload()
        self.pipe.enable_vae_tiling()
        self.pipe.enable_attention_slicing(1)
        
        self.pipe.to("cuda")
        
    def infer(self, inputs):
        """Генерация видео с валидацией"""
        # Валидация входных данных
        prompt = inputs.get("prompt", "").strip()
        if not prompt:
            raise ValueError("Prompt cannot be empty")
            
        negative_prompt = inputs.get("negative_prompt", 
            "Bright tones, overexposed, static, blurred details, worst quality, low quality")
        
        # Параметры из успешных примеров[4]
        height = inputs.get("height", 480)  # Начинаем с 480P для стабильности
        width = inputs.get("width", 480)
        num_frames = inputs.get("num_frames", 81)  # 5 секунд при 16 FPS
        
        # Валидация разрешения
        if height not in [480, 720] or width not in [480, 720]:
            raise ValueError("Only 480x480 and 720x720 supported")
            
        try:
            # Генерация с правильными параметрами[2][4]
            output = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_frames=num_frames,
                num_inference_steps=30,  # Минимум 30 для качества
                guidance_scale=5.0,      # Проверенное значение
                generator=torch.Generator().manual_seed(42)
            )
            
            # Сохранение видео
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
                export_to_video(output.frames[0], temp_file.name, fps=16)
                
                # Кодируем в base64 для возврата
                with open(temp_file.name, "rb") as video_file:
                    video_data = base64.b64encode(video_file.read()).decode()
                
                os.unlink(temp_file.name)
                
            return {
                "video_base64": video_data,
                "num_frames": num_frames,
                "resolution": f"{width}x{height}"
            }
            
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            raise RuntimeError("GPU memory exceeded. Try lower resolution or fewer frames")
            
    def finalize(self):
        """Очистка ресурсов"""
        if hasattr(self, 'pipe'):
            del self.pipe
        torch.cuda.empty_cache()
