import json
import torch
import torch.nn.functional as F
import imageio
from PIL import Image
import io
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from autoencoder import Autoencoder
from sklearn.neighbors import NearestNeighbors
import os


class Frame:
    def __init__(self, points, prompt, layer_idx, tops, neighbors):
        self.points = points
        self.prompt = prompt
        self.layer_idx = layer_idx
        self.tops = tops
        self.neighbors = neighbors

    def tojson(self):
        return {
            'points': self.points.tolist(),
            'prompt': self.prompt,
            'layer_idx': self.layer_idx,
            'tops': self.tops,
            'neighbors': self.neighbors
        }



class Visualizer:
    def __init__(self, prompts, model_name='unsloth/Llama-3.2-1B'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        torch.set_default_tensor_type(torch.cuda.FloatTensor if self.device.type == 'cuda' else torch.FloatTensor)
        torch.set_float32_matmul_precision('medium')
        self.model_name = model_name
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.top_k = 5
        self.prompts = prompts
        self.frames = []
        self.states = []

    def norm(self, state):
        if self.model_name == 'microsoft/phi-2':
            return self.model.model.final_layernorm(state)
        elif self.model_name.startswith('openai-community/gpt2'):
            return self.model.transformer.ln_f(state)
        elif self.model_name == 'EleutherAI/pythia-410m':
            return self.model.gpt_neox.final_layer_norm(state)
        else:
            return self.model.model.norm(state)

    def forward_toks(self, toks):
        output = self.model.forward(toks, output_hidden_states=True)
        states = self.get_normed_states(output)
        # print(toks, self.tokenizer.decode(toks[0, :]))
        self.states.append((states, self.tokenizer.decode(toks[0, :])))

    def forward(self):
        for prompt in self.prompts:
            enc = self.tokenizer(prompt, return_tensors='pt', return_attention_mask=False)
            input_ids = enc['input_ids'].to(self.device)
            tok_len = input_ids.shape[1]
            for i in range(1, tok_len + 1):
                self.forward_toks(input_ids[:, :i])

    def get_normed_states(self, output):
        hidden_states = output.hidden_states
        normed_states = [self.norm(hs).detach() for hs in hidden_states[:-1]]
        final = hidden_states[-1]
        normed_states.append(final)
        return normed_states

    def load_autoencoder(self, filename='weights/checkpoint.pth'):
        input_dim = self.states[0][0][0][0][0].size()[0]
        n_layers = len(self.states[0][0])
        self.autoencoder = Autoencoder(
            input_dim=input_dim,
        ).to(self.device)
        self.autoencoder.load_state_dict(torch.load(filename))
        self.autoencoder.eval()
        print(f"Autoencoder weights loaded from {filename}")

    def encode(self):
        print('embedding', self.states[0][0][-1][0].shape)
        n_layers = len(self.states[0][0])
        target_layer = n_layers // 2
        self.embeddings = []
        for (state, prompt) in self.states:
            layer = state[target_layer]
            points = self.autoencoder(layer.float().to(self.device))[0][0]
            tops = self.top_tokens(layer)
            neighbors = self.find_nearest_neighbors(points)
            frame = Frame(points, prompt, target_layer, tops, neighbors)
            self.frames.append(frame)

    def logits(self, state):
        state = state.unsqueeze(0)
        if self.model_name == 'EleutherAI/pythia-410m':
            return self.model.embed_out(state).float()
        else:
            return self.model.lm_head(state).float()

    def logprobs(self, state):
        logits = self.logits(state)
        return F.softmax(logits[0], dim=-1)

    def top_tokens(self, state):
        top_probs, top_indices = torch.topk(self.logprobs(state)[0, -1, :], self.top_k)
        return [(self.tokenizer.decode([idx]), top_probs[j].item()) for j, idx in enumerate(top_indices)]

    def find_nearest_neighbors(self, points, n_neighbors=4):
        points = points.cpu().detach().numpy()
        neighbors = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
        neighbors.fit(points)
        distances, indices = neighbors.kneighbors(points)
        return [list(zip(indices[i].tolist(), distances[i].tolist()))[1:] for i in range(len(distances))]

    def visualize(self):
        data = json.dumps({
            'frames': [frame.tojson() for frame in self.frames],
            'n_layers': len(self.states[0][0])
        })
        with open('visualize_template.html', 'r') as template_file:
            template = template_file.read()
            modified = template.replace('$$DATA$$', data)
            with open('visualize.html', 'w') as out_file:
                out_file.write(modified)
                print('wrote modified template')

    def visualize_2d(self):
        frame_size = 512
        scaled_embeddings = []
        for layer_embeddings in self.embeddings:
            min_vals = layer_embeddings.min(dim=0, keepdim=True).values
            max_vals = layer_embeddings.max(dim=0, keepdim=True).values
            scaled = ((layer_embeddings - min_vals) / (max_vals - min_vals)) * (frame_size - 1)
            scaled_embeddings.append(scaled.cpu().detach().numpy().astype(int))
        print('visualizing', min_vals, max_vals)
        images = []
        for idx, layer_points in enumerate(scaled_embeddings):
            img = np.zeros((frame_size, frame_size, 3), dtype=np.uint8)
            for pidx, point in enumerate(layer_points):
                pn = pidx / len(layer_points)
                x, y = point
                img[y, x] = (255, 255, 255)
            image = Image.fromarray(img)
            images.append(image)
        video_path = "layer_animation.mp4"
        imageio.mimsave(video_path, images, format='MP4', fps=4)
        print(f"Animation saved to {video_path}")

    def run(self):
        self.forward()
        self.load_autoencoder()
        self.encode()
        self.visualize()

if __name__ == '__main__':
    prompts = [
        "If personality is an unbroken series of successful gestures, then there was something gorgeous about him, some heightened sensitivity to the promises of life"         
        ]
    visualizer = Visualizer(prompts)
    visualizer.run()
