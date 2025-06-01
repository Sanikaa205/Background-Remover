from flask import Flask, render_template, request, jsonify, Response, send_from_directory
from rembg import remove, new_session
from PIL import Image
import os
import cv2
import numpy as np
import uuid
from werkzeug.utils import secure_filename
from datetime import datetime
import subprocess
import threading

app = Flask(__name__, static_folder="static", template_folder="templates")

# Configuration
UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"
BACKGROUNDS_FOLDER = "static/backgrounds"
RECORDINGS_FOLDER = "static/recordings"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(BACKGROUNDS_FOLDER, exist_ok=True)
os.makedirs(RECORDINGS_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'mp4'}
session = new_session("u2net")

# Video settings
frame_width, frame_height = 640, 480
target_fps = 30  # Target frame rate
recording = False
video_writer = None
current_background = None
background_image = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def finalize_video(file_path):
    """Ensure MP4 file is properly encoded with correct colors and speed"""
    if os.path.exists(file_path):
        try:
            temp_file = file_path + ".temp.mp4"
            subprocess.run([
                'ffmpeg', '-y', '-i', file_path,
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-movflags', '+faststart',
                '-pix_fmt', 'yuv420p',
                '-filter:v', 'setpts=1.0*PTS',  # Fix playback speed
                '-vsync', '0',                  # Prevent frame duplication
                '-r', str(target_fps),          # Force output frame rate
                '-color_primaries', 'bt709',    # Proper color encoding
                '-color_trc', 'bt709',
                '-colorspace', 'bt709',
                temp_file
            ], check=True)
            os.replace(temp_file, file_path)
        except Exception as e:
            print(f"Video finalization failed (falling back to original): {str(e)}")

def process_frame(frame):
    """Process frame with proper color handling"""
    # Convert from BGR to RGB (OpenCV to standard)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    if background_image is not None:
        # Convert to RGBA for background removal
        removed_bg = remove(frame_rgb, session=session)
        
        # Prepare background - resize to match frame size
        bg_resized = cv2.resize(background_image, (frame.shape[1], frame.shape[0]))
        
        # Convert to numpy array if it's a PIL Image
        if isinstance(removed_bg, Image.Image):
            removed_bg = np.array(removed_bg)
        
        # Create mask from alpha channel
        mask = removed_bg[:, :, 3] / 255.0
        mask = cv2.merge([mask, mask, mask])
        
        # Convert foreground to RGB
        foreground = cv2.cvtColor(removed_bg, cv2.COLOR_RGBA2RGB)
        
        # Composite the image
        frame_rgb = (foreground * mask + bg_resized * (1 - mask)).astype(np.uint8)
    
    return frame_rgb

def generate_frames():
    global recording, video_writer
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    cap.set(cv2.CAP_PROP_FPS, target_fps)
    
    # Try to get actual camera FPS
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    if actual_fps <= 0:
        actual_fps = target_fps
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        # Process frame (returns RGB format)
        processed_frame = process_frame(frame)
        
        # Recording logic
        if recording:
            if video_writer is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = os.path.join(RECORDINGS_FOLDER, f'recording_{timestamp}.mp4')
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(
                    output_file,
                    fourcc,
                    actual_fps,
                    (frame_width, frame_height)
                )
            
            # Convert back to BGR for video writer
            video_writer.write(cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR))
        
        # Stream frame (already in RGB format)
        _, buffer = cv2.imencode('.jpg', processed_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    # Cleanup
    if video_writer is not None:
        video_writer.release()
    cap.release()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/img_bg')
def img_bg():
    return render_template('img_bg.html')

@app.route('/live_vd')
def live_vd():
    return render_template('live_vd.html')

@app.route('/replace')
def replace():
    return render_template('replace.html')

@app.route('/img_rep')
def img_rep():
    return render_template('img_rep.html')

@app.route('/replace-image-bg', methods=['POST'])
def replace_image_bg():
    if 'image' not in request.files or 'background' not in request.files:
        return jsonify({'error': 'Both image and background files are required'}), 400
    
    image_file = request.files['image']
    bg_file = request.files['background']
    
    if image_file.filename == '' or bg_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not allowed_file(image_file.filename) or not allowed_file(bg_file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    try:
        # Process foreground image
        foreground = Image.open(image_file.stream).convert('RGBA')
        foreground.thumbnail((1024, 1024))
        foreground = remove(foreground, session=session)
        
        # Process background image
        background = Image.open(bg_file.stream).convert('RGB')
        background = background.resize(foreground.size)
        
        # Composite images
        composite = Image.new('RGB', foreground.size)
        composite.paste(background, (0, 0))
        composite.paste(foreground, (0, 0), foreground)
        
        # Save output
        output_filename = f'replaced_bg_{uuid.uuid4().hex[:8]}.jpg'
        output_path = os.path.join(PROCESSED_FOLDER, output_filename)
        composite.save(output_path, quality=95)
        
        return send_from_directory(PROCESSED_FOLDER, output_filename)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/remove-bg', methods=['POST'])
def remove_bg():
    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    try:
        input_img = Image.open(file.stream).convert('RGBA')
        input_img.thumbnail((1024, 1024))
        output = remove(input_img, session=session)
        
        output_filename = f'no_bg_{uuid.uuid4().hex[:8]}.png'
        output_path = os.path.join(PROCESSED_FOLDER, output_filename)
        output.save(output_path)
        
        return send_from_directory(PROCESSED_FOLDER, output_filename, as_attachment=True)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def load_background(bg_name):
    if bg_name:
        bg_path = os.path.join(BACKGROUNDS_FOLDER, bg_name)
        if os.path.exists(bg_path):
            img = cv2.imread(bg_path)
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    return None

@app.route('/set-background', methods=['POST'])
def set_bg():
    global current_background, background_image
    data = request.get_json()
    current_background = data.get('background')
    background_image = load_background(current_background)
    return jsonify({'status': 'success', 'background': current_background})

@app.route('/start_recording', methods=['POST'])
def start_recording():
    global recording
    if not recording:
        recording = True
        return jsonify({'status': 'success', 'message': 'Recording started'})
    return jsonify({'status': 'already_recording'})

@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    global recording, video_writer
    if recording:
        recording = False
        if video_writer is not None:
            video_writer.release()
            video_writer = None
            recordings = sorted(os.listdir(RECORDINGS_FOLDER), reverse=True)
            if recordings:
                # Finalize video in a separate thread to avoid blocking
                threading.Thread(target=finalize_video, args=(os.path.join(RECORDINGS_FOLDER, recordings[0]),)).start()
        return jsonify({'status': 'success', 'message': 'Recording stopped'})
    return jsonify({'status': 'not_recording'})

@app.route('/list_recordings', methods=['GET'])
def list_recordings():
    recordings = []
    for filename in sorted(os.listdir(RECORDINGS_FOLDER), reverse=True):
        if filename.endswith('.mp4'):
            filepath = os.path.join(RECORDINGS_FOLDER, filename)
            size = os.path.getsize(filepath) / (1024 * 1024)  # Size in MB
            recordings.append({
                'name': filename,
                'url': f'/static/recordings/{filename}',
                'size': f'{size:.2f} MB',
                'date': datetime.fromtimestamp(os.path.getctime(filepath)).strftime('%Y-%m-%d %H:%M:%S')
            })
    return jsonify({'recordings': recordings})

@app.route('/delete_recording/<filename>', methods=['DELETE'])
def delete_recording(filename):
    try:
        filepath = os.path.join(RECORDINGS_FOLDER, secure_filename(filename))
        if os.path.exists(filepath):
            os.remove(filepath)
            return jsonify({'status': 'success', 'message': 'Recording deleted'})
        return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/upload-bg', methods=['POST'])
def upload_bg():
    if 'custom_bg' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['custom_bg']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        filename = secure_filename(f"{uuid.uuid4().hex[:8]}_{file.filename}")
        filepath = os.path.join(BACKGROUNDS_FOLDER, filename)
        
        img = Image.open(file.stream)
        img.thumbnail((1920, 1080))  # Resize to max 1080p
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img.save(filepath)
        
        return jsonify({
            'status': 'success',
            'filename': filename,
            'url': f'/static/backgrounds/{filename}'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, threaded=True)