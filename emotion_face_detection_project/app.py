import os
import uuid
import boto3
import datetime
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from your_script import process_video

app = Flask(__name__)

# AWS Configuration
AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')
DYNAMODB_TABLE = os.getenv('DYNAMODB_TABLE', 'emotion-face-recognition')
S3_BUCKET = os.getenv('S3_BUCKET', 'my-emotion-model-bucket')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'mov', 'avi'}

# Initialize AWS clients
dynamodb = boto3.resource('dynamodb', region_name=AWS_REGION)
s3 = boto3.client('s3', region_name=AWS_REGION)
table = dynamodb.Table(DYNAMODB_TABLE)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_photo():
    try:
        person_id = request.form.get('person_id')
        if not person_id:
            return jsonify({'error': 'person_id is required'}), 400

        file = request.files.get('file')
        if not file:
            return jsonify({'error': 'No file uploaded'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed'}), 400

        # Generate unique filename
        file_ext = file.filename.rsplit('.', 1)[1].lower()
        filename = f"{person_id}_{str(uuid.uuid4())[:8]}.{file_ext}"
        s3_key = f"uploads/{filename}"

        # Upload to S3
        s3.upload_fileobj(
            file,
            S3_BUCKET,
            s3_key,
            ExtraArgs={'ContentType': file.content_type}
        )

        # Store metadata in DynamoDB
        item = {
            'person_id': person_id,
            'file_id': str(uuid.uuid4()),
            's3_key': s3_key,
            'file_type': file_ext,
            'upload_time': str(datetime.datetime.now()),
            'processed': False
        }
        
        table.put_item(Item=item)

        return jsonify({
            'success': True,
            'message': 'File uploaded successfully',
            's3_path': f"s3://{S3_BUCKET}/{s3_key}",
            'file_type': file_ext
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/process-video', methods=['POST'])
def process_video_endpoint():
    try:
        video_s3_key = request.form.get('video_s3_key')
        if not video_s3_key:
            return jsonify({'error': 'video_s3_key is required'}), 400

        # Generate output path
        output_key = f"processed/{os.path.basename(video_s3_key)}_emotion.mp4"

        # Process the video
        process_video(
            input_bucket=S3_BUCKET,
            input_key=video_s3_key,
            output_bucket=S3_BUCKET,
            output_key=output_key,
            aws_region=AWS_REGION
        )

        return jsonify({
            'success': True,
            'message': 'Video processing started',
            'output_path': f"s3://{S3_BUCKET}/{output_key}"
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/videos', methods=['GET'])
def list_videos():
    try:
        # Get all video files from DynamoDB
        response = table.scan(
            FilterExpression='file_type IN (:mp4, :mov, :avi)',
            ExpressionAttributeValues={
                ':mp4': 'mp4',
                ':mov': 'mov',
                ':avi': 'avi'
            }
        )
        
        videos = response.get('Items', [])
        return jsonify({'videos': videos})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
