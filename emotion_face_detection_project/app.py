import os
import uuid
import boto3
import datetime
import logging
from functools import wraps
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from your_script import process_video

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
class Config:
    AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')
    DYNAMODB_TABLE = os.getenv('DYNAMODB_TABLE', 'KnownFaces')
    S3_BUCKET = os.getenv('S3_BUCKET', 'my-emotion-model-bucket')
    INPUT_PREFIX = 'input/'
    OUTPUT_PREFIX = 'output/'
    ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi'}
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB limit

app.config.from_object(Config)

# Error handler decorator
def handle_errors(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {f.__name__}: {str(e)}", exc_info=True)
            return jsonify({'error': str(e)}), 500
    return wrapper

# Initialize AWS clients
try:
    s3 = boto3.client('s3', region_name=Config.AWS_REGION)
    dynamodb = boto3.resource('dynamodb', region_name=Config.AWS_REGION)
    table = dynamodb.Table(Config.DYNAMODB_TABLE)
    logger.info("AWS clients initialized successfully")
except Exception as e:
    logger.error(f"AWS client initialization failed: {e}")
    raise

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
@handle_errors
def upload_video():
    """Handle video uploads to input folder"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if not allowed_file(file.filename):
        return jsonify({'error': f'Only {", ".join(Config.ALLOWED_EXTENSIONS)} files allowed'}), 400

    # Generate S3 key
    filename = f"{uuid.uuid4().hex[:8]}_{secure_filename(file.filename)}"
    s3_key = f"{Config.INPUT_PREFIX}{filename}"

    # Upload to S3
    s3.upload_fileobj(
        file,
        Config.S3_BUCKET,
        s3_key,
        ExtraArgs={
            'ContentType': file.content_type,
            'Metadata': {
                'original_name': file.filename,
                'upload_time': datetime.datetime.now().isoformat()
            }
        }
    )
    logger.info(f"Uploaded to s3://{Config.S3_BUCKET}/{s3_key}")

    return jsonify({
        'status': 'success',
        's3_key': s3_key,
        'message': 'Video uploaded for processing',
        'download_url': f"/download/{s3_key}"
    })

@app.route('/process', methods=['POST'])
@handle_errors
def process_video_endpoint():
    """Trigger video processing"""
    data = request.get_json()
    if not data or 'input_key' not in data:
        return jsonify({'error': 'input_key is required'}), 400

    if not data['input_key'].startswith(Config.INPUT_PREFIX):
        return jsonify({'error': 'Invalid input key format'}), 400

    result = process_video(
        input_bucket=Config.S3_BUCKET,
        input_key=data['input_key'],
        output_bucket=Config.S3_BUCKET,
        output_key=f"{Config.OUTPUT_PREFIX}{os.path.basename(data['input_key'])}_processed.mp4",
        aws_region=Config.AWS_REGION
    )

    if result.get('status') == 'error':
        return jsonify(result), 400
        
    return jsonify(result)

@app.route('/videos', methods=['GET'])
@handle_errors
def list_processed_videos():
    """List processed videos with pagination"""
    continuation_token = request.args.get('continuation_token')
    
    list_args = {
        'Bucket': Config.S3_BUCKET,
        'Prefix': Config.OUTPUT_PREFIX,
        'MaxKeys': 50
    }
    
    if continuation_token:
        list_args['ContinuationToken'] = continuation_token

    response = s3.list_objects_v2(**list_args)
    
    videos = [{
        'key': obj['Key'],
        'size': obj['Size'],
        'last_modified': obj['LastModified'].isoformat(),
        'url': f"/download/{obj['Key']}"
    } for obj in response.get('Contents', [])]
    
    result = {
        'videos': videos,
        'count': len(videos)
    }
    
    if 'NextContinuationToken' in response:
        result['continuation_token'] = response['NextContinuationToken']
    
    return jsonify(result)

@app.route('/download/<path:s3_key>', methods=['GET'])
@handle_errors
def download_file(s3_key):
    """Generate presigned URL for download"""
    url = s3.generate_presigned_url(
        'get_object',
        Params={
            'Bucket': Config.S3_BUCKET,
            'Key': s3_key
        },
        ExpiresIn=3600
    )
    return jsonify({'url': url})

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.datetime.now().isoformat(),
        'aws_region': Config.AWS_REGION
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
