import os
import openai
from flask import Flask, request, jsonify

app = Flask(__name__)

# Configure OpenAI API
openai.api_key = os.getenv('OPENAI_API_KEY')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {'role': 'user', 'content': user_message}
            ]
        )
        chat_response = response['choices'][0]['message']['content']
        return jsonify({'response': chat_response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    # Process the uploaded file
    file.save(os.path.join('uploads', file.filename))
    return jsonify({'message': 'File uploaded successfully'}), 200

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)  # Create upload directory if it doesn't exist
    app.run(debug=True, host='0.0.0.0', port=5000)