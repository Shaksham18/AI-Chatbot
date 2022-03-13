import random
import json
import torch
from model import NeuralNet
from nltk_utils import NltkUtils


class ChatCore:
    def __init__(self, bot_name, intent_file, model_file):
        self.bot_name = bot_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        with open(intent_file, 'r') as f:
            self.intents = json.load(f)

        FILE = model_file
        data = torch.load(FILE)

        self.input_size = data['input_size']
        self.hidden_size = data['hidden_size']
        self.output_size = data['output_size']
        self.all_words = data['all_words']
        self.tags = data['tags']
        self.model_state = data['model_state']

        self.model = NeuralNet(self.input_size, self.hidden_size, self.output_size)
        self.model.load_state_dict(self.model_state)
        self.model.eval()
        self.nltk = NltkUtils()

    def get_response(self, msg):
        sentence = self.nltk.tokenize(msg)
        X = self.nltk.bag_of_words(sentence, self.all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X)
        output = self.model(X)
        _, predicted = torch.max(output, dim=1)
        tag = self.tags[predicted.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]

        if prob.item() > 0.7:
            for intent in self.intents["intents"]:
                if tag == intent['tag']:
                    return random.choice(intent['responses'])
        else:
            return "I Do not understand..."
