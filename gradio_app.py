import os
import gradio as gr
import subprocess
from PIL import Image
import tempfile
from pathlib import Path
import re
import threading
import queue
import queue
from inference import main as inference_main 

def check_image_ratio(image_path):
    """Check if image has 1:1 aspect ratio"""
    with Image.open(image_path) as img:
        width, height = img.size
        return abs(width - height) < 10  # Allow small deviation from perfect square

def make_square_image(input_path, output_path):
    """Convert image to 1:1 aspect ratio using FFmpeg"""
    cmd = [
        'ffmpeg', '-i', input_path,
        '-vf', 'pad=max(iw\\,ih):max(iw\\,ih):(ow-iw)/2:(oh-ih)/2:color=white',
        '-y', output_path
    ]
    subprocess.run(cmd, check=True)
    return output_path

def process_output(pipe, q):
    """Process subprocess output and extract progress information"""
    pattern = r'(\d+)%\|'  # Pattern to match percentage in tqdm output

    while True:
        line = pipe.readline()
        if not line:  
            break  # Stop when there’s no more output

        print(line.strip())  # ✅ Print everything for debugging

        if '%|' in line:  # Progress updates
            match = re.search(pattern, line)
            if match:
                progress = int(match.group(1))
                q.put(('progress', progress))
        else:
            q.put(('message', line.strip()))  # Send other messages to queue

import subprocess
import os

def run_inference(image, audio, allow_non_square, progress=gr.Progress()):
    if not image or not audio:
        return None, "Please provide both image and audio files."
    
    # Create temporary directory for processing
    temp_dir = tempfile.mkdtemp()
    
    # Save uploaded files to temp directory
    image_path = os.path.join(temp_dir, "input_image" + os.path.splitext(image)[1])
    audio_path = os.path.join(temp_dir, "input_audio" + os.path.splitext(audio)[1])
    
    os.replace(image, image_path)
    os.replace(audio, audio_path)
    
    # Create output directory
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Call inference directly instead of using subprocess
        output_video_path = inference_main(
            image_path=image_path,
            audio_path=audio_path,
            output_dir=output_dir,
            config="configs/inference.yaml"
        )
        return output_video_path, "Generation completed successfully!"
        os.system('rd /s /q "output/audio_preprocess"')
    
    except Exception as e:
        return None, f"Error during generation: {str(e)}"



# Create Gradio interface
with gr.Blocks(title="MEMO Video Generation") as demo:
    gr.Markdown("# MEMO: Motion-Driven Emotional Talking Face Generation")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Input Image", type="filepath")
            input_audio = gr.Audio(label="Input Audio", type="filepath")
            allow_non_square = gr.Checkbox(
                label="Allow non-square images", 
                value=False,
                info="Check this to skip 1:1 aspect ratio conversion"
            )
            generate_btn = gr.Button("Generate Video", variant="primary")
            
        with gr.Column():
            output_video = gr.Video(label="Generated Video")
            status_text = gr.Textbox(label="Status", interactive=False)
    
    generate_btn.click(
        fn=run_inference,
        inputs=[input_image, input_audio, allow_non_square],
        outputs=[output_video, status_text]
    )

if __name__ == "__main__":
    demo.launch()
