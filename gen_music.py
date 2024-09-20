import torch
import numpy as np
import soundfile as sf
from transformers import AutoProcessor, MusicgenForConditionalGeneration

class GenMusic:
    def __init__(self, model_name="facebook/musicgen-large"):
        self.model_name = model_name
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        # Load the model and move it to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MusicgenForConditionalGeneration.from_pretrained(model_name).to(self.device)
    
    def do(self, prompt, play_time):
        # Convert play_time to tokens
        tokens_per_second = 256 / 30  # Assuming 256 tokens correspond to 30 seconds of audio
        max_new_tokens = int(tokens_per_second * play_time)
        
        # Ensure max_new_tokens is within a reasonable range
        max_new_tokens = min(max_new_tokens, 1024)  # Arbitrary upper limit for safety
        
        # Process the prompt
        inputs = self.processor(text=[prompt], padding=True, return_tensors="pt").to(self.device)
        
        # Generate the music
        try:
            audio_values = self.model.generate(
                **inputs, 
                do_sample=True, 
                guidance_scale=3, 
                max_new_tokens=max_new_tokens
            )
        except IndexError as e:
            print(f"Error generating music: {e}")
            return None, None
        
        # Convert the generated audio to a numpy array
        sampling_rate = self.model.config.audio_encoder.sampling_rate
        audio_array = audio_values[0].cpu().numpy()
        
        # Normalize audio to the 16-bit signed PCM range (-32768 to 32767)
        audio_array = audio_array / np.max(np.abs(audio_array))  # Normalize to [-1, 1]
        audio_array = np.int16(audio_array * 32767)  # Scale to [-32768, 32767]
        
        # Save the generated audio to a .wav file using soundfile, explicitly specifying format and subtype
        sf.write("generated_music.wav", audio_array, sampling_rate, format='WAV', subtype='PCM_16')
        
        return audio_array, sampling_rate
