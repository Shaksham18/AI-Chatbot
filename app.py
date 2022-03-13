# using flask_restful
from flask import Flask
from flask_restful import Api
from resource import Hello, GetBotResponse

app = Flask(__name__)
api = Api(app)

# APIs
api.add_resource(GetBotResponse, '/getBtResp')


# driver function
if __name__ == '__main__':
    app.run(debug=True)
