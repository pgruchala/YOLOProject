import cv2
import torch
from flask import Flask, render_template, Response, request, redirect, url_for, flash
from ultralytics import YOLO
import time
import os
from werkzeug.utils import secure_filename
from flask import send_from_directory

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'avi', 'mov'}
app.secret_key = 'supersecretkey'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

try:
    model = YOLO('yolov11Emotions/weights/best.pt')
except Exception as e:
    print(f"Błąd podczas ładowania modelu YOLO: {e}")
    model = None

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def generate_frames():
    if model is None:
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + 'Błąd: Model nie załadowany.' + b'\r\n')
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + 'Błąd: Nie można otworzyć kamery.' + b'\r\n')
        return

    prev_time = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            break

        current_time = time.time()
        fps = 1 / (current_time - prev_time) if prev_time > 0 else 0
        prev_time = current_time

        results = model(frame, stream=True, conf=0.35)

        for r in results:
            annotated_frame = r.plot()
            cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/webcam_feed')
def webcam_feed():
    return render_template('webcam.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload', methods=['POST'])
def upload_file():
    if model is None:
        flash('Błąd: Model AI nie załadowany. Spróbuj ponownie później.')
        return redirect(request.url)

    if 'file' not in request.files:
        flash('Brak pliku w żądaniu!')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('Nie wybrano pliku!')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        processed_filename = "processed_" + filename
        processed_filepath = os.path.join(app.config['UPLOAD_FOLDER'], processed_filename)

        if filename.rsplit('.', 1)[1].lower() in {'mp4', 'avi', 'mov'}:
            cap = cv2.VideoCapture(filepath)
            if not cap.isOpened():
                flash(f"Nie można otworzyć pliku wideo: {filename}")
                return redirect(url_for('index'))


            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(processed_filepath, fourcc, fps, (width, height))

            if not out.isOpened():
                flash(f"Błąd: Nie można otworzyć pliku wyjściowego wideo do zapisu: {processed_filepath}")
                cap.release()
                return redirect(url_for('index'))

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                results = model(frame)
                annotated_frame = results[0].plot()
                out.write(annotated_frame)
            
            cap.release()
            out.release()
            flash(f'Film "{filename}" przetworzony!')
            return redirect(url_for('display_processed', filename=processed_filename, filetype='video'))

        else:
            try:
                img = cv2.imread(filepath)
                if img is None:
                    flash(f"Nie można wczytać obrazu: {filename}")
                    return redirect(url_for('index'))
                results = model(img)
                annotated_img = results[0].plot()
                
                cv2.imwrite(processed_filepath, annotated_img)
                flash(f'Obraz "{filename}" przetworzony!')
                return redirect(url_for('display_processed', filename=processed_filename, filetype='image'))

            except Exception as e:
                flash(f"Błąd podczas przetwarzania obrazu: {e}")
        return redirect(url_for('index'))
    else:
        flash('Nieprawidłowy typ pliku. Dozwolone typy: png, jpg, jpeg, gif, mp4, avi, mov.')
        return redirect(request.url)

@app.route('/display_processed/<filename>')
def display_processed(filename):
    file_extension = filename.rsplit('.', 1)[1].lower()
    if file_extension in {'mp4', 'avi', 'mov'}:
        file_type = 'video'
    else:
        file_type = 'image'
    return render_template('display_processed.html', filename=filename, file_type=file_type)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    # Determine the MIME type based on file extension
    file_extension = filename.rsplit('.', 1)[1].lower()
    if file_extension in {'mp4', 'mov'}:
        mimetype = 'video/mp4'
    elif file_extension == 'avi':
        mimetype = 'video/x-msvideo' # AVI MIME type
    elif file_extension in {'png', 'jpg', 'jpeg', 'gif'}:
        mimetype = f'image/{file_extension}'
    else:
        mimetype = None # Let send_from_directory infer

    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, mimetype=mimetype)


if __name__ == '__main__':
    app.run(debug=True)