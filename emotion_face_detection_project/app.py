import os
import uuid
import boto3
import datetime
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from your_script import process_video
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# AWS Configuration (matches your_script.py)
AWS_CONFIG = {
    'region': 'us-east-1',
    'dynamodb_table': 'KnownFaces',
    's3_bucket': 'my-emotion-model-bucket',
    'input_prefix': 'input/',
    'output_prefix': 'output/'
}

# Initialize AWS clients
try:
    s3 = boto3.client('s3', region_name=AWS_CONFIG['region'])
    dynamodb = boto3.resource('dynamodb', region_name=AWS_CONFIG['region'])
    table = dynamodb.Table(AWS_CONFIG['dynamodb_table'])
except Exception as e:
    logger.error(f"AWS client initialization failed: {e}")
    raise

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'mp4', 'mov', 'avi'}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    """Handle video uploads to input folder"""
    try:
        # Validate file
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
            
        if not allowed_file(file.filename):
            return jsonify({'error': 'Only video files allowed'}), 400

        # Generate S3 key for input folder
        filename = f"{uuid.uuid4().hex[:8]}_{secure_filename(file.filename)}"
        s3_key = f"{AWS_CONFIG['input_prefix']}{filename}"

        # Upload to S3 input folder
        s3.upload_fileobj(
            file,
            AWS_CONFIG['s3_bucket'],
            s3_key,
            ExtraArgs={
                'ContentType': file.content_type,
                'Metadata': {
                    'original_name': file.filename
                }
            }
        )
        logger.info(f"Uploaded to s3://{AWS_CONFIG['s3_bucket']}/{s3_key}")

        return jsonify({
            'status': 'success',
            's3_key': s3_key,
            'message': 'Video uploaded for processing'
        })

    except Exception as e:
        logger.error(f"Upload failed: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/process', methods=['POST'])
def process_video_endpoint():
    """Trigger video processing"""
    try:
        data = request.get_json()
        input_key = data.get('input_key')
        
        if not input_key:
            return jsonify({'error': 'input_key is required'}), 400

        # Process video (matches your_script.py interface)
        result = process_video(
            input_bucket=AWS_CONFIG['s3_bucket'],
            input_key=input_key,
            output_bucket=AWS_CONFIG['s3_bucket'],
            output_key=None,  # Let your_script.py generate output path
            aws_region=AWS_CONFIG['region']
        )

        if result.get('status') == 'error':
            return jsonify(result), 500
            
        return jsonify(result)

    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/videos', methods=['GET'])
def list_processed_videos():
    """List processed videos from output folder"""
    try:
        response = s3.list_objects_v2(
            Bucket=AWS_CONFIG['s3_bucket'],
            Prefix=AWS_CONFIG['output_prefix']
        )
        
        videos = [{
            'key': obj['Key'],
            'size': obj['Size'],
            'last_modified': obj['LastModified'].isoformat()
        } for obj in response.get('Contents', [])]
        
        return jsonify({'videos': videos})
    
    except Exception as e:
        logger.error(f"Failed to list videos: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
