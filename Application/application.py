from generate import Slm

slm = Slm()

# print(slm.generate("I feel lonely and anxious sometimes because"))

from flask import Flask, request, jsonify, render_template
app = Flask(__name__)



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_text():
    data = request.json
    input_text = data.get('input_text', '')
    max_new_tokens = data.get('max_new_tokens', 40)

    generated_text = slm.generate(input_text, max_new_tokens=max_new_tokens)
    return jsonify({'generated_text': generated_text})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)