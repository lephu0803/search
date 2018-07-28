from flask import Flask, request, jsonify, Response
import base64
import json
from search import Find_Image

app = Flask(__name__)
x = Find_Image()
# root
@app.route("/")
def index():
    """
    this is a root dir of my server
    :return: str
    """
    return "This is root!!!!"


# GET
@app.route('/image/<image>')
def hello_image(image):
    """
    this serves as a demo purpose
    :param image:
    :return: str
    """
    return "Hello %s!" % image


# POST
@app.route('/api/post_some_data', methods=['POST'])
def get_text_prediction():
    """
    predicts requested text whether it is ham or spam
    :return: json
    """
    json = request.get_json()
    # print(json)
    if len(json['image']) == 0:
        return jsonify({'error': 'invalid input'})
    imgdata = base64.b64decode(json['image'])
    filename = 'some_image.png'  # I assume you have a way of picking unique filenames
    with open(filename, 'wb') as f:
        f.write(imgdata)
    idx, dis= x.search(querry_image = 'some_image.png')
    print(idx, dis)
    # print(idx)
    data_path = x.find(idx)
    json_results = x.return_json(data_path)
    print(data_path)
    # print(json_results)
    # json_results = jsonify(json_results)
    return jsonify(json_results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
