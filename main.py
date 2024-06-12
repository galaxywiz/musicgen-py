# pip install -r requirements.txt 
# 갱신시 pip freeze > requirements.txt

import warnings
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from IPython.display import Audio
import scipy.io.wavfile
import numpy as np

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, message="torch.nn.utils.weight_norm is deprecated")
warnings.filterwarnings("ignore", category=UserWarning, message="To copy construct from a tensor")

# Load the processor and model
model_name = "facebook/musicgen-small"
processor = AutoProcessor.from_pretrained(model_name)
model = MusicgenForConditionalGeneration.from_pretrained(model_name)

# Define your text prompt
text_prompt = ["Make music that sounds like you'd hear in a village shop in a medieval fantasy style"]

# Process the text prompt
inputs = processor(text=text_prompt, padding=True, return_tensors="pt")

# Generate music based on the text prompt
def generate_music(inputs, total_length, chunk_length_tokens):
    audio_chunks = []
    num_chunks = total_length // (chunk_length_tokens / (model.config.audio_encoder.sampling_rate // 1000))
    for i in range(int(num_chunks)):
        print(f"Generating chunk {i + 1}/{int(num_chunks)}")
        audio_values = model.generate(**inputs, do_sample=True, guidance_scale=3, max_new_tokens=chunk_length_tokens)
        if audio_values is not None:
            audio_chunks.append(audio_values[0, 0].numpy())
        else:
            print("No audio generated for this chunk.")
    if audio_chunks:
        return np.concatenate(audio_chunks)
    else:
        print("No audio data was generated.")
        return None

# Define the total length and chunk length in seconds
total_length = 600  # 10 minutes
chunk_length = 40  # Generate in 40-second chunks

# Convert chunk length to the equivalent in tokens
chunk_length_tokens = chunk_length * (model.config.audio_encoder.sampling_rate // 1000)

# Generate the music
audio_output = generate_music(inputs, total_length, chunk_length_tokens)

# Save the generated music to a file if audio_output is not None
if audio_output is not None:
    sampling_rate = model.config.audio_encoder.sampling_rate
    scipy.io.wavfile.write("musicgen_output_fantasy.wav", rate=sampling_rate, data=audio_output)
    # Listen to the generated music
    Audio(audio_output, rate=sampling_rate)
else:
    print("Failed to generate audio.")
