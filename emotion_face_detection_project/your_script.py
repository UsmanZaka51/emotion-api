import os
import ast
import cv2
import boto3
import numpy as np
import face_recognition
from fer import FER
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmotionVideoProcessor:
    def __init__(self):
        self.region = 'us-east-1'
        self.s3_bucket = 'my-emotion-model-bucket'
        self.input_prefix = 'input/'
        self.output_prefix = 'output/'

        self.dynamodb = boto3.resource('dynamodb', region_name=self.region)
        self.s3 = boto3.client('s3', region_name=self.region)
        self.emotion_detector = FER(mtcnn=True)

        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        cv2.setUseOptimized(True)
        cv2.setNumThreads(4)

    def _get_face_encodings(self):
        table = self.dynamodb.Table('KnownFaces')
        response = table.scan(ProjectionExpression="person_id,embedding")

        known_faces = {'names': [], 'encodings': []}
        for item in response.get('Items', []):
            try:
                encoding = np.array(ast.literal_eval(item['embedding']), dtype=np.float32)
                known_faces['encodings'].append(encoding)
                known_faces['names'].append(item['person_id'])
            except Exception as e:
                logger.warning(f"Invalid encoding for {item.get('person_id')}: {e}")
        if not known_faces['encodings']:
            raise ValueError("No face encodings available in DynamoDB")
        return known_faces

    def _annotate_frame(self, frame, known_faces):
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(known_faces['encodings'], encoding, tolerance=0.6)
                name = "Unknown"

                if True in matches:
                    distances = face_recognition.face_distance(known_faces['encodings'], encoding)
                    best_match = np.argmin(distances)
                    name = known_faces['names'][best_match]

                face_roi = frame[top:bottom, left:right]
                try:
                    emotion, score = self.emotion_detector.top_emotion(face_roi)
                    label = f"{name}: {emotion or 'Neutral'}"
                    if score:
                        label += f" ({score:.2f})"
                except Exception:
                    label = f"{name}: Neutral"

                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            return frame
        except Exception as e:
            logger.error(f"Frame error: {e}")
            return frame

    def process_video(self, input_key):
        if not input_key.startswith(self.input_prefix):
            input_key = self.input_prefix + input_key
        output_key = self.output_prefix + os.path.basename(input_key) + "_processed.mp4"

        temp_input = f"/tmp/{os.path.basename(input_key)}"
        temp_output = f"/tmp/{os.path.basename(output_key)}"

        try:
            self.s3.download_file(self.s3_bucket, input_key, temp_input)
            logger.info(f"Downloaded {input_key}")

            known_faces = self._get_face_encodings()
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
                processed = self._annotate_frame(frame, known_faces)
                out.write(processed)
                frame_count += 1
                if frame_count % 100 == 0:
                    logger.info(f"Processed {frame_count} frames")

            cap.release()
            out.release()

            self.s3.upload_file(temp_output, self.s3_bucket, output_key)
            logger.info(f"Uploaded to {output_key}")

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
            for f in [temp_input, temp_output]:
                if os.path.exists(f):
                    try:
                        os.remove(f)
                    except:
                        pass

# Global instance
video_processor = EmotionVideoProcessor()

def process_video(input_bucket, input_key, output_bucket, output_key, aws_region='us-east-1'):
    if input_bucket != 'my-emotion-model-bucket' or output_bucket != 'my-emotion-model-bucket':
        raise ValueError("Bucket name must be 'my-emotion-model-bucket'")
    return video_processor.process_video(input_key)
