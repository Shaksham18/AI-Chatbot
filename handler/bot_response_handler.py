from flask import jsonify
from core.chat_core import ChatCore


class BotResponseHandler:
    def __init__(self):
        pass

    def get_response(self, msg):
        obj = ChatCore("Sam", intent_file="../core/ml-models/intent.json", model_file="../core/ml-models/data.pth")
        response = obj.get_response(msg)
        return jsonify({'data': response})
