from flask import Flask, request
from flask_cors import CORS, cross_origin

# Init server
app = Flask(__name__)

# Apply CORS
CORS(app)
app.config['COR_HEADERS'] = 'config'

@app.route("/add", methods=['POST', 'GET'])
@cross_origin(origin='*')
def add():
    return str(int(request.form.get('a')) + int(request.form.get('b')))


# Start Server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port='6868')