from transformers import AutoProcessor, MusicgenForConditionalGeneration
import scipy.io.wavfile

class GenMusic:
    def __init__(self, model_name="facebook/musicgen-large"):
        self.model_name = model_name
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = MusicgenForConditionalGeneration.from_pretrained(model_name)
    
    def do(self, prompt, play_time):
        # Convert play_time to tokens
        # Assuming 256 tokens roughly correspond to 30 seconds of audio
        tokens_per_second = 256 / 30
        max_new_tokens = int(tokens_per_second * play_time)
        
        # Process the prompt
        inputs = self.processor(text=[prompt], padding=True, return_tensors="pt")
        
        # Generate the music
        audio_values = self.model.generate(
            **inputs, 
            do_sample=True, 
            guidance_scale=3, 
            max_new_tokens=max_new_tokens
        )
        
        # Convert the generated audio to a numpy array
        sampling_rate = self.model.config.audio_encoder.sampling_rate
        audio_array = audio_values[0].numpy()
        
        # Save the generated audio to a .wav file
        scipy.io.wavfile.write("generated_music.wav", rate=sampling_rate, data=audio_array)
        
        return audio_array, sampling_rate

# # Example usage:
# gen_music = GenMusic()
# audio_array, sampling_rate = gen_music.do("80s pop track with bassy drums and synth", 60)

# # Optionally, play the generated audio in a Jupyter notebook
# from IPython.display import Audio
# Audio(audio_array, rate=sampling_rate)
