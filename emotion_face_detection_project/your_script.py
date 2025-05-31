import os
import ast
import cv2
import boto3
import numpy as np
import face_recognition
from fer import FER
from datetime import datetime
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Optional optimizations
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
cv2.setUseOptimized(True)
cv2.setNumThreads(4)

def get_aws_clients(region_name='us-east-1'):
    """Initialize AWS clients with error handling"""
    try:
        return {
            's3': boto3.client('s3', region_name=region_name),
            'dynamodb': boto3.resource('dynamodb', region_name=region_name)
        }
    except Exception as e:
        logger.error(f"AWS client initialization failed: {e}")
        raise

def load_known_faces_from_dynamodb(table_name="KnownFaces", region_name='us-east-1'):
    """Load face encodings from DynamoDB with retry logic"""
    clients = get_aws_clients(region_name)
    table = clients['dynamodb'].Table(table_name)
    
    known_data = {'names': [], 'encodings': []}
    try:
        response = table.scan(ProjectionExpression="person_id,embedding")
        for item in response.get('Items', []):
            try:
                known_data['names'].append(item['person_id'])
                known_data['encodings'].append(np.array(ast.literal_eval(item['embedding']), dtype=np.float32))
            except (ValueError, KeyError) as e:
                logger.warning(f"Invalid face data for {item.get('person_id')}: {e}")
        
        if not known_data['encodings']:
            logger.warning("No valid face encodings found in DynamoDB")
        
        return known_data
    except Exception as e:
        logger.error(f"DynamoDB scan failed: {e}")
        raise

def handle_s3_file(bucket_name, s3_key, local_path, operation='download', region_name='us-east-1'):
    """Generic S3 file handler for up/downloads"""
    clients = get_aws_clients(region_name)
    try:
        if operation == 'download':
            clients['s3'].download_file(bucket_name, s3_key, local_path)
            logger.info(f"Downloaded {s3_key} to {local_path}")
        elif operation == 'upload':
            clients['s3'].upload_file(local_path, bucket_name, s3_key)
            logger.info(f"Uploaded {local_path} to {s3_key}")
        else:
            raise ValueError("Invalid operation - must be 'download' or 'upload'")
        return True
    except Exception as e:
        logger.error(f"S3 {operation} failed: {e}")
        raise

def process_video_frame(frame, known_data, emotion_detector):
    """Process a single frame for faces and emotions"""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
        # Face recognition
        name = "Unknown"
        matches = face_recognition.compare_faces(known_data['encodings'], encoding, tolerance=0.6)
        if True in matches:
            best_match_idx = np.argmin(face_recognition.face_distance(known_data['encodings'], encoding))
            name = known_data['names'][best_match_idx]

        # Emotion detection
        face_crop = frame[top:bottom, left:right]
        try:
            top_emotion, score = emotion_detector.top_emotion(face_crop)
            emotion_label = f"{top_emotion} ({score:.2f})" if top_emotion else "Neutral"
        except Exception as e:
            logger.warning(f"Emotion detection failed: {e}")
            emotion_label = "Neutral"

        # Annotate frame
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, emotion_label, (left, bottom + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    return frame

def process_video(
    s3_input_bucket,
    s3_input_key,
    s3_output_bucket=None,
    s3_output_key=None,
    dynamo_table="KnownFaces",
    region_name='us-east-1'
):
    """Main video processing pipeline with enhanced error handling"""
    # File paths
    local_input = f"/tmp/{os.path.basename(s3_input_key)}"
    local_output = f"/tmp/processed_{os.path.basename(s3_input_key)}"
    
    try:
        logger.info(f"Starting processing for {s3_input_bucket}/{s3_input_key}")
        
        # 1. Download video
        handle_s3_file(s3_input_bucket, s3_input_key, local_input, 'download', region_name)
        
        # 2. Load known faces
        known_data = load_known_faces_from_dynamodb(dynamo_table, region_name)
        emotion_detector = FER(mtcnn=True)
        
        # 3. Process video
        cap = cv2.VideoCapture(local_input)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {local_input}")
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(local_output, fourcc, fps, (width, height))
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            processed_frame = process_video_frame(frame, known_data, emotion_detector)
            out.write(processed_frame)
            frame_count += 1
            
            if frame_count % 100 == 0:
                logger.info(f"Processed {frame_count} frames")
        
        cap.release()
        out.release()
        logger.info(f"Finished processing {frame_count} frames")
        
        # 4. Upload result
        if s3_output_bucket and s3_output_key:
            handle_s3_file(s3_output_bucket, s3_output_key, local_output, 'upload', region_name)
        
        return {
            'status': 'success',
            'processed_frames': frame_count,
            'output_path': local_output,
            's3_output_path': f"s3://{s3_output_bucket}/{s3_output_key}" if s3_output_bucket else None
        }
        
    except Exception as e:
        logger.error(f"Video processing failed: {e}")
        return {
            'status': 'error',
            'error': str(e),
            'processed_frames': frame_count if 'frame_count' in locals() else 0
        }
    finally:
        # Cleanup
        for f in [local_input, local_output]:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except Exception as e:
                    logger.warning(f"Could not delete {f}: {e}")
