from flask import jsonify, request
from flask_restful import Resource

from handler.bot_response_handler import BotResponseHandler


class GetBotResponse(Resource):

    def post(self):
        data = request.get_json()  # status code
        obj = BotResponseHandler()
        return obj.get_response(data['msg'])
