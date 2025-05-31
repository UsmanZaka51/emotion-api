from flask import Flask, request, render_template, jsonify
import os
import boto3
import face_recognition
import uuid
import numpy as np
from your_script import process_video

app = Flask(__name__)

# AWS configuration
REGION = 'us-east-1'
BUCKET = 'my-emotion-model-bucket'
DYNAMODB_TABLE = 'KnownFaces'

s3 = boto3.client('s3', region_name=REGION)
dynamodb = boto3.resource('dynamodb', region_name=REGION)
table = dynamodb.Table(DYNAMODB_TABLE)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['POST'])
def register():
    name = request.form['name']
    image_file = request.files['face']

    image_np = face_recognition.load_image_file(image_file)
    encodings = face_recognition.face_encodings(image_np)

    if not encodings:
        return jsonify({'status': 'error', 'message': 'No face detected'}), 400

    face_encoding = encodings[0].tolist()
    table.put_item(Item={
        'person_id': name,
        'embedding': str(face_encoding)
    })

    return jsonify({'status': 'success', 'message': f'Registered face for {name}'}), 200

@app.route('/upload', methods=['POST'])
def upload():
    if 'video' not in request.files:
        return jsonify({'status': 'error', 'message': 'No video uploaded'}), 400

    video_file = request.files['video']
    filename = f"{uuid.uuid4()}.mp4"
    s3_key = f"input/{filename}"

    try:
        s3.upload_fileobj(video_file, BUCKET, s3_key)

        result = process_video(
            input_bucket=BUCKET,
            input_key=s3_key,
            output_bucket=BUCKET,
            output_key=f"output/{filename}_processed.mp4"
        )

        return jsonify(result), 200 if result['status'] == 'success' else 500

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
