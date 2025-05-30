import os
import ast
import cv2
import boto3
import numpy as np
import face_recognition
from fer import FER

# Optional speed optimization
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
cv2.setUseOptimized(True)
cv2.setNumThreads(4)

def load_known_faces_from_dynamodb(table_name="KnownFaces"):
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table(table_name)

    try:
        response = table.scan(ProjectionExpression="person_id,embedding")
        items = response.get('Items', [])

        known_names = []
        known_encodings = []
        for item in items:
            name = item['person_id']
            embedding = np.array(ast.literal_eval(item['embedding']), dtype=np.float32)
            known_names.append(name)
            known_encodings.append(embedding)

        return known_names, known_encodings
    except Exception as e:
        print(f"Failed to load from DynamoDB: {e}")
        return [], []

def download_file_from_s3(bucket_name, s3_key, local_path):
    s3 = boto3.client('s3')
    s3.download_file(bucket_name, s3_key, local_path)

def upload_file_to_s3(local_path, bucket_name, s3_key):
    s3 = boto3.client('s3')
    s3.upload_file(local_path, bucket_name, s3_key)

def process_video(s3_input_bucket, s3_input_key, s3_output_bucket=None, s3_output_key=None, dynamo_table="KnownFaces"):
    local_input_path = "/tmp/input_video.mp4"
    local_output_path = "/tmp/output_with_emotions.mp4"

    # Download input video from S3
    download_file_from_s3(s3_input_bucket, s3_input_key, local_input_path)

    known_names, known_encodings = load_known_faces_from_dynamodb(dynamo_table)
    if not known_encodings:
        raise RuntimeError("No known faces found in DynamoDB.")

    emotion_detector = FER(mtcnn=True)
    cap = cv2.VideoCapture(local_input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {local_input_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(local_output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
            name = "Unknown"
            matches = face_recognition.compare_faces(known_encodings, encoding, tolerance=0.6)
            if True in matches:
                best_match_index = np.argmin(face_recognition.face_distance(known_encodings, encoding))
                name = known_names[best_match_index]

            face_crop = frame[top:bottom, left:right]
            top_emotion, score = emotion_detector.top_emotion(face_crop)
            emotion_label = f"{top_emotion} ({score:.2f})" if top_emotion else "Neutral"

            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame, emotion_label, (left, bottom + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        out.write(frame)

    cap.release()
    out.release()

    # Upload result video to S3 if bucket and key are provided
    if s3_output_bucket and s3_output_key:
        upload_file_to_s3(local_output_path, s3_output_bucket, s3_output_key)

    return local_output_path
