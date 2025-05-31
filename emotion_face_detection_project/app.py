import os
import uuid
import boto3
import datetime
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from your_script import process_video
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# AWS Configuration
AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')
DYNAMODB_TABLE = os.getenv('DYNAMODB_TABLE', 'emotion-face-recognition')
S3_BUCKET = os.getenv('S3_BUCKET', 'my-emotion-model-bucket')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'mov', 'avi'}

# Initialize AWS clients with error handling
try:
    dynamodb = boto3.resource('dynamodb', region_name=AWS_REGION)
    s3 = boto3.client('s3', region_name=AWS_REGION)
    rekognition = boto3.client('rekognition', region_name=AWS_REGION)
    table = dynamodb.Table(DYNAMODB_TABLE)
except Exception as e:
    logger.error(f"AWS client initialization failed: {e}")
    raise

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        # Validate inputs
        person_id = request.form.get('person_id', 'anonymous')
        file = request.files.get('file')
        
        if not file or file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed'}), 400

        # Generate secure filename
        file_ext = file.filename.rsplit('.', 1)[1].lower()
        filename = f"{person_id}_{uuid.uuid4().hex[:8]}.{file_ext}"
        s3_key = f"uploads/{filename}"

        # Upload to S3
        try:
            s3.upload_fileobj(
                file,
                S3_BUCKET,
                s3_key,
                ExtraArgs={
                    'ContentType': file.content_type,
                    'Metadata': {
                        'person_id': person_id,
                        'original_filename': secure_filename(file.filename)
                    }
                }
            )
            logger.info(f"File uploaded to S3: {s3_key}")
        except Exception as upload_error:
            logger.error(f"S3 upload failed: {upload_error}")
            return jsonify({'error': 'File upload failed'}), 500

        # Store metadata in DynamoDB
        item = {
            'person_id': person_id,
            'file_id': str(uuid.uuid4()),
            's3_key': s3_key,
            'file_type': file_ext,
            'upload_time': datetime.datetime.now().isoformat(),
            'processed': False,
            'file_size': request.content_length
        }

        try:
            table.put_item(Item=item)
            logger.info(f"Metadata stored in DynamoDB: {item['file_id']}")
        except Exception as db_error:
            logger.error(f"DynamoDB put failed: {db_error}")
            # Attempt to delete the uploaded file if DB fails
            s3.delete_object(Bucket=S3_BUCKET, Key=s3_key)
            return jsonify({'error': 'Metadata storage failed'}), 500

        return jsonify({
            'success': True,
            's3_key': s3_key,
            'file_id': item['file_id'],
            'person_id': person_id
        })

    except Exception as e:
        logger.error(f"Upload failed: {str(e)}", exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/process-video', methods=['POST'])
def process_video_endpoint():
    try:
        data = request.get_json() if request.is_json else request.form
        video_s3_key = data.get('video_s3_key')
        
        if not video_s3_key:
            return jsonify({'error': 'video_s3_key is required'}), 400

        # Generate output path
        base_name = os.path.basename(video_s3_key)
        output_key = f"processed/{os.path.splitext(base_name)[0]}_emotion.mp4"

        # Process the video using your_script.py
        result = process_video(
            s3_input_bucket=S3_BUCKET,
            s3_input_key=video_s3_key,
            s3_output_bucket=S3_BUCKET,
            s3_output_key=output_key,
            region_name=AWS_REGION
        )

        if result.get('status') == 'error':
            return jsonify({'error': result.get('error', 'Processing failed')}), 500

        # Update DynamoDB record
        try:
            table.update_item(
                Key={'s3_key': video_s3_key},
                UpdateExpression='SET processed = :val, processed_time = :time, output_key = :out',
                ExpressionAttributeValues={
                    ':val': True,
                    ':time': datetime.datetime.now().isoformat(),
                    ':out': output_key
                }
            )
        except Exception as db_error:
            logger.error(f"Failed to update DynamoDB: {db_error}")

        return jsonify({
            'success': True,
            'output_path': f"s3://{S3_BUCKET}/{output_key}",
            'processed_frames': result.get('processed_frames', 0)
        })

    except Exception as e:
        logger.error(f"Video processing failed: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/videos', methods=['GET'])
def list_videos():
    try:
        # Get paginated results
        last_evaluated_key = request.args.get('last_key')
        scan_args = {
            'FilterExpression': 'file_type IN (:mp4, :mov, :avi)',
            'ExpressionAttributeValues': {
                ':mp4': 'mp4',
                ':mov': 'mov',
                ':avi': 'avi'
            },
            'Limit': 50
        }
        
        if last_evaluated_key:
            scan_args['ExclusiveStartKey'] = ast.literal_eval(last_evaluated_key)

        response = table.scan(**scan_args)
        
        return jsonify({
            'videos': response.get('Items', []),
            'last_evaluated_key': response.get('LastEvaluatedKey')
        })
    
    except Exception as e:
        logger.error(f"Failed to list videos: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/video/<file_id>', methods=['GET'])
def get_video_details(file_id):
    try:
        response = table.get_item(Key={'file_id': file_id})
        item = response.get('Item')
        
        if not item:
            return jsonify({'error': 'Video not found'}), 404
            
        # Generate pre-signed URL for easy access
        if item['file_type'] in ['mp4', 'mov', 'avi']:
            url = s3.generate_presigned_url(
                'get_object',
                Params={'Bucket': S3_BUCKET, 'Key': item['s3_key']},
                ExpiresIn=3600
            )
            item['preview_url'] = url
            
        return jsonify(item)
    
    except Exception as e:
        logger.error(f"Failed to get video details: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
