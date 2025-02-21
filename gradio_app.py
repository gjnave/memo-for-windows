import os
import gradio as gr
import subprocess
from PIL import Image
import tempfile
from pathlib import Path
import re
import threading
import queue

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
            break
        if b'%|' in line:  # tqdm progress line
            match = re.search(pattern, line.decode())
            if match:
                progress = int(match.group(1))
                q.put(('progress', progress))
        else:
            # Regular output line
            q.put(('message', line.decode().strip()))

def run_inference(image, audio, allow_non_square, progress=gr.Progress()):
    if not image or not audio:
        return None, "Please provide both image and audio files."
    
    # Define progress stages and their corresponding weights
    stages = {
        "starting": 0.1,  # 10% of total progress
        "image_processing": 0.1,  # 10% of total progress
        "generation": 0.7,  # 70% of total progress
        "finalizing": 0.1,  # 10% of total progress
    }
    
    # Start progress
    progress(0, desc="Starting...")
    
    # Create temporary directory for processing
    temp_dir = tempfile.mkdtemp()
    
    # Save uploaded files to temp directory
    image_path = os.path.join(temp_dir, "input_image" + os.path.splitext(image)[1])
    audio_path = os.path.join(temp_dir, "input_audio" + os.path.splitext(audio)[1])
    
    os.replace(image, image_path)
    os.replace(audio, audio_path)
    
    # Check image ratio and process if needed
    if not allow_non_square and not check_image_ratio(image_path):
        progress(stages["starting"], desc="Processing image to square format...")
        try:
            square_image_path = os.path.join(temp_dir, "square_input.png")
            make_square_image(image_path, square_image_path)
            image_path = square_image_path
        except subprocess.CalledProcessError as e:
            return None, f"Error processing image: {str(e)}"
    
    # Create output directory
    output_dir = os.path.join(temp_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    
    progress(stages["starting"] + stages["image_processing"], desc="Starting MEMO generation...")
    
    # Run MEMO inference
    try:
        cmd = [
            "python", "inference.py",
            "--config", "configs/inference.yaml",
            "--input_image", image_path,
            "--input_audio", audio_path,
            "--output_dir", output_dir
        ]
        
        # Create process with pipe for output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=False
        )
        
        # Create queue for communication between threads
        q = queue.Queue()
        
        # Start thread to process output
        thread = threading.Thread(target=process_output, args=(process.stdout, q))
        thread.daemon = True
        thread.start()
        
        # Process queue messages and update progress
        current_progress = stages["starting"] + stages["image_processing"]
        while process.poll() is None or not q.empty():
            try:
                msg_type, msg = q.get_nowait()
                if msg_type == 'progress':
                    # Scale progress within the generation stage (20% - 90%)
                    generation_progress = (msg / 100) * stages["generation"]
                    overall_progress = stages["starting"] + stages["image_processing"] + generation_progress
                    progress(overall_progress)  # Update progress without text
                elif msg_type == 'message':
                    # Ignore messages for now (optional: log them if needed)
                    pass
            except queue.Empty:
                continue
        
        # Check process return code
        if process.returncode != 0:
            return None, "Error during generation"
        
        # Finalize progress
        progress(1.0, desc="Complete!")
        
        # Find output video
        output_files = list(Path(output_dir).glob("*.mp4"))
        if not output_files:
            return None, "No output video generated"
            
        output_video = str(output_files[0])
        return output_video, "Generation completed successfully!"
        
    except subprocess.CalledProcessError as e:
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