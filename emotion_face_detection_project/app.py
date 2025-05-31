from flask import Flask, request, jsonify, render_template
from your_script import process_video

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    data = request.form
    input_bucket = data.get('input_bucket')
    input_key = data.get('input_key')
    output_bucket = data.get('output_bucket')
    output_key = data.get('output_key')
    aws_region = data.get('aws_region', 'us-east-1')

    if not all([input_bucket, input_key, output_bucket, output_key]):
        return jsonify({'status': 'error', 'message': 'Missing required parameters'}), 400

    try:
        process_video(input_bucket, input_key, output_bucket, output_key)
        s3_url = f"https://{output_bucket}.s3.amazonaws.com/{output_key}"
        return jsonify({'status': 'success', 'output_url': s3_url})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
