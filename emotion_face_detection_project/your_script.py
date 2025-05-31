import os
import ast
import cv2
import boto3
import numpy as np
import face_recognition
from fer import FER
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EmotionVideoProcessor:
    def __init__(self):
        """Initialize with AWS us-east-1 region and S3 bucket structure"""
        self.region = 'us-east-1'
        self.s3_bucket = 'my-emotion-model-bucket'
        self.input_prefix = 'input/'
        self.output_prefix = 'output/'
        
        # Initialize AWS clients
        self.dynamodb = boto3.resource('dynamodb', region_name=self.region)
        self.s3 = boto3.client('s3', region_name=self.region)
        self.rekognition = boto3.client('rekognition', region_name=self.region)
        
        # Initialize models
        self.emotion_detector = FER(mtcnn=True)
        
        # Performance optimizations
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
        cv2.setUseOptimized(True)
        cv2.setNumThreads(4)

    def _get_face_encodings(self):
        """Load face encodings from DynamoDB (person_id + embedding)"""
        try:
            table = self.dynamodb.Table('KnownFaces')
            response = table.scan(ProjectionExpression="person_id,embedding")
            
            known_faces = {'names': [], 'encodings': []}
            
            for item in response.get('Items', []):
                try:
                    # Convert string embedding to numpy array
                    encoding = np.array(ast.literal_eval(item['embedding']), dtype=np.float32)
                    known_faces['encodings'].append(encoding)
                    known_faces['names'].append(item['person_id'])
                except (ValueError, SyntaxError) as e:
                    logger.warning(f"Invalid encoding for {item.get('person_id')}: {e}")
            
            if not known_faces['encodings']:
                logger.error("No valid face encodings found in DynamoDB")
                raise ValueError("No face encodings available")
            
            return known_faces
            
        except Exception as e:
            logger.error(f"DynamoDB access failed: {e}")
            raise

    def _annotate_frame(self, frame, known_faces):
        """Process a single frame with face recognition and emotion detection"""
        try:
            # Convert to RGB (required by face_recognition)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
                # Recognize face
                matches = face_recognition.compare_faces(
                    known_faces['encodings'], 
                    encoding,
                    tolerance=0.6
                )
                
                name = "Unknown"
                if True in matches:
                    face_distances = face_recognition.face_distance(
                        known_faces['encodings'],
                        encoding
                    )
                    best_match_idx = np.argmin(face_distances)
                    name = known_faces['names'][best_match_idx]

                # Detect emotion
                face_roi = frame[top:bottom, left:right]
                try:
                    emotion, score = self.emotion_detector.top_emotion(face_roi)
                    label = f"{name}: {emotion or 'Neutral'}"
                    if emotion and score:
                        label += f" ({score:.2f})"
                except Exception as e:
                    logger.warning(f"Emotion detection failed: {e}")
                    label = f"{name}: Neutral"

                # Draw annotations
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.putText(frame, label, (left, top-15), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            return frame
            
        except Exception as e:
            logger.error(f"Frame processing failed: {e}")
            return frame

    def process_video(self, input_key):
        """Complete video processing pipeline"""
        # Validate input path
        if not input_key.startswith(self.input_prefix):
            input_key = f"{self.input_prefix}{input_key}"
        
        # Generate output path
        output_key = f"{self.output_prefix}{os.path.basename(input_key)}_processed.mp4"
        
        # Local temporary files
        temp_input = f"/tmp/{os.path.basename(input_key)}"
        temp_output = f"/tmp/{os.path.basename(output_key)}"
        
        try:
            logger.info(f"Starting processing: s3://{self.s3_bucket}/{input_key}")
            
            # 1. Download video from S3 input folder
            self.s3.download_file(self.s3_bucket, input_key, temp_input)
            logger.info(f"Downloaded to: {temp_input}")
            
            # 2. Load known faces
            known_faces = self._get_face_encodings()
            logger.info(f"Loaded {len(known_faces['encodings'])} face encodings")
            
            # 3. Process video
            cap = cv2.VideoCapture(temp_input)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))

            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                processed_frame = self._annotate_frame(frame, known_faces)
                out.write(processed_frame)
                frame_count += 1
                
                if frame_count % 100 == 0:
                    logger.info(f"Processed {frame_count} frames")

            cap.release()
            out.release()
            logger.info(f"Finished processing {frame_count} frames")
            
            # 4. Upload to S3 output folder
            self.s3.upload_file(temp_output, self.s3_bucket, output_key)
            logger.info(f"Uploaded result to: s3://{self.s3_bucket}/{output_key}")
            
            return {
                'status': 'success',
                'input_path': f"s3://{self.s3_bucket}/{input_key}",
                'output_path': f"s3://{self.s3_bucket}/{output_key}",
                'processed_frames': frame_count
            }
            
        except Exception as e:
            logger.error(f"Processing failed: {e}", exc_info=True)
            return {
                'status': 'error',
                'error': str(e),
                'input_path': f"s3://{self.s3_bucket}/{input_key}"
            }
            
        finally:
            # Cleanup temporary files
            for f in [temp_input, temp_output]:
                if os.path.exists(f):
                    try:
                        os.remove(f)
                    except Exception as e:
                        logger.warning(f"Couldn't delete {f}: {e}")

# Global instance for Flask integration
video_processor = EmotionVideoProcessor()

def process_video(input_bucket, input_key, output_bucket, output_key, aws_region='us-east-1'):
    """Flask-compatible interface"""
    # Validate bucket consistency
    if input_bucket != 'my-emotion-model-bucket' or output_bucket != 'my-emotion-model-bucket':
        raise ValueError("Bucket name must be 'my-emotion-model-bucket'")
    
    return video_processor.process_video(input_key)
