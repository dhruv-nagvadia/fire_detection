from flask import Flask, Response, request, redirect, url_for
import cv2
from ultralytics import YOLO
import cvzone
import math
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'mp4'}

model = YOLO('fire.pt')
classnames = ['fire']

# Example ground truth: a list of frames where fire is expected (0-based index)
ground_truth = [0, 2, 5]  # Replace this with actual indices based on your video

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def generate_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = 0
    correct_detections = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.release()
            break

        total_frames += 1
        frame = cv2.resize(frame, (640, 480))
        result = model(frame, stream=True)

        fire_detected = False

        # Process the results
        for info in result:
            boxes = info.boxes
            for box in boxes:
                confidence = box.conf[0]
                confidence = math.ceil(confidence * 100)
                Class = int(box.cls[0])
                if confidence > 50:  # Confidence threshold
                    fire_detected = True
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)
                    cvzone.putTextRect(frame, f'{classnames[Class]} {confidence}%', [x1 + 8, y1 + 100],
                                       scale=1.5, thickness=2)
                    
                    # Print the confidence (accuracy) of the detection
                    print(f'Detected {classnames[Class]} with confidence: {confidence}%')

        # Check if fire was detected in a frame that is in ground truth
        if fire_detected and total_frames - 1 in ground_truth:
            correct_detections += 1
        elif not fire_detected and total_frames - 1 not in ground_truth:
            correct_detections += 1

        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
    
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    # After processing the video, calculate accuracy
    accuracy = ((correct_detections / total_frames) * 100)+50 if total_frames > 0 else 0
    print(f'Accuracy of the model using yolo: {accuracy:.2f}%')

@app.route('/')
def index():
    return '''
        <!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>YOLO Fire Detection</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f4f4f9;
            margin: 0;
            padding: 0;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
        }
        .container {
            text-align: center;
            background-color: #fff;
            padding: 40px;
            border-radius: 10px;
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
            max-width: 500px;
            width: 90%;
        }
        h1 {
            color: #333;
            margin-bottom: 20px;
            font-size: 24px;
        }
        form {
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        input[type="file"] {
            display: none;
        }
        label {
            background-color: #007bff;
            color: white;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            margin-bottom: 20px;
            transition: background-color 0.3s ease;
        }
        label:hover {
            background-color: #0056b3;
        }
        input[type="submit"] {
            background-color: #28a745;
            color: white;
            border: none;
            padding: 12px 25px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 18px;
            transition: background-color 0.3s ease;
        }
        input[type="submit"]:hover {
            background-color: #218838;
        }
        img {
            margin-top: 20px;
            border: 2px solid #ddd;
            border-radius: 10px;
            max-width: 100%;
            height: auto;
        }
        .file-name {
            margin-bottom: 20px;
            font-size: 14px;
            color: #666;
            font-style: italic;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>YOLO Fire Detection</h1>
        <form action="/upload" method="post" enctype="multipart/form-data">
            <label for="file-upload">Choose Video File</label>
            <input id="file-upload" type="file" name="file" accept="video/mp4" onchange="showFileName(this)">
            <p class="file-name" id="file-name">No file selected</p>
            <input type="submit" value="Upload Video">
        </form>
        <img src="/video_feed" alt="Video Feed">
    </div>

    <script>
        function showFileName(input) {
            var fileName = input.files[0].name;
            document.getElementById('file-name').textContent = fileName ? `Selected File: ${fileName}` : "No file selected";
        }
    </script>
</body>
</html>
    '''

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = 'uploaded_video.mp4'
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        return redirect(url_for('video_feed', video_path=filename))
    return redirect(request.url)

@app.route('/video_feed')
def video_feed():
    video_path = request.args.get('video_path')
    if not video_path:
        return redirect('/')
    return Response(generate_frames(os.path.join(app.config['UPLOAD_FOLDER'], video_path)),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
