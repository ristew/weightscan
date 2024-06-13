import datetime
import os
import math
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, BitsAndBytesConfig
from autoencoder import Autoencoder

class Scan():
    def __init__(self, prompts):
        # Determine if CUDA is available, otherwise use CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        torch.set_default_tensor_type(torch.cuda.FloatTensor if self.device.type == 'cuda' else torch.FloatTensor)
        torch.set_float32_matmul_precision('medium')

        self.model_name = 'openai-community/gpt2'
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, trust_remote_code=True).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.top_k = 5
        self.prompts = prompts
        self.states = []

    def norm(self, state):
        if self.model_name == 'microsoft/phi-2':
            return self.model.model.final_layernorm(state)
        elif self.model_name == 'openai-community/gpt2':
            return self.model.transformer.ln_f(state)
        else:
            return self.model.model.norm(state)

    def forward(self):
        for prompt in self.prompts:
            enc = self.tokenizer(prompt, return_tensors='pt', return_attention_mask=False)
            input_ids = enc['input_ids'].to(self.device)
            output = self.model.forward(input_ids, output_hidden_states=True)
            states = self.get_normed_states(output)
            self.states.append(states)

        self.autoencoder = Autoencoder(
            input_dim=self.states[0][0][0][0].size()[0],
            compressed_dim=(2048, 3),
            temporal_weight=1e4,
            distance_weight=1e-3,
            lr=1e-4,
            num_epochs=3,
            training_set=self.states,
            logprob_fn=self.logprobs,
        ).to(self.device)
        self.embeddings = self.autoencode()

    def get_normed_states(self, output):
        hidden_states = output.hidden_states
        print('hidden state shape', hidden_states[0][0].shape)
        normed_states = [self.norm(hs) for hs in hidden_states[:-1]]
        final = hidden_states[-1]
        normed_states.append(final)
        return normed_states

    def autoencode(self):
        self.autoencoder.train()
        res = [self.autoencoder(n.float().to(self.device))[0][0] for n in self.states[0]]
        return res

    def logits(self, state):
        return self.model.lm_head(state.unsqueeze(0)).float()

    def logprobs(self, state):
        logits = self.logits(state)
        return F.softmax(logits[0], dim=-1)

    def top_tokens(self, state):
        top_probs, top_indices = torch.topk(self.logprobs(state)[0, -1, :], self.top_k)
        return [(self.tokenizer.decode([idx]), top_probs[j].item()) for j, idx in enumerate(top_indices)]

    def find_nearest_neighbors(self, embeddings, n_neighbors=4):
        nn = []
        for layer in embeddings:
            neighbors = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
            neighbors.fit(layer)
            distances, indices = neighbors.kneighbors(layer)
            nn.append([list(zip(indices[i].tolist(), distances[i].tolist()))[1:] for i in range(len(distances))])
        return nn

    def visualize(self):
        points = [(p * 16).tolist() for p in self.embeddings]
        tops = [self.top_tokens(self.states[0][i]) for i in range(len(self.embeddings))]
        nn_indices = self.find_nearest_neighbors(points)

        data = json.dumps({
            'points': points,
            'tops': tops,
            'prompt': self.prompts[0],
            'neighbors': nn_indices
        })
        with open('visualize_template.html', 'r') as template_file:
            template = template_file.read()
            modified = template.replace('$$POINTS$$', data)
            with open('visualize.html', 'w') as out_file:
                out_file.write(modified)
                print('wrote modified template')

    def test(self):
        self.forward()
        self.visualize()

if __name__ == '__main__':
    Scan([
        'The sun rises in the east and sets in the',
        'When I go fishing, I like to think about',
        '22 * 8 =',
        'It is going to be okay',
        'How are you today?',
        'The quick brown fox jumps over the lazy dog',
        'What is the capital of France?',
        'I love the smell of coffee in the morning',
        'The square root of 144 is',
        'I am feeling a bit under the weather today',
        'Can you recommend a good book to read?',
        '42 / 6 =',
        'I am excited about my upcoming vacation',
        'What is your favorite color?',
        'The early bird catches the worm',
        'How many planets are in our solar system?',
        'I enjoy listening to classical music',
        '15 + 27 =',
        'I am grateful for my family and friends',
        'What is the weather forecast for tomorrow?',
        'Honesty is the best policy',
        '64 - 19 =',
        'I am learning to play the guitar',
        'What is the largest continent in the world?'
    ]).test()
