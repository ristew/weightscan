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

        self.model_name = 'openai-community/gpt2-medium'
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, trust_remote_code=True, quantization_config=quantization_config)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.top_k = 5
        self.prompts = prompts
        self.states = []

    def norm(self, state):
        if self.model_name == 'microsoft/phi-2':
            return self.model.model.final_layernorm(state)
        elif self.model_name.startswith('openai-community/gpt2'):
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
            compressed_dim=(4096, 3),
            temporal_weight=4e3,
            lr=2e-5,
            weight_decay=0.001,
            num_epochs=2,
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
        self.autoencoder.train_set()
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
        points = [(p * 64).tolist() for p in self.embeddings]
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
        'After traveling north, we turned left. Several hours later, we turned left again. Then one more time. We were now traveling',
        'North and south, the eternal poles that guide our navigation. These cardinal directions have shaped human exploration and settlement patterns for millennia. From the frozen tundras of the Arctic to the vast savannas of Africa, the north-south axis has defined climates, cultures, and migratory routes. In many mythologies, the north is associated with mystery and magic, while the south often represents warmth and',
        'The sun rises in the east and sets in the west, a daily cosmic dance that has captivated humanity since time immemorial. This celestial journey marks the passage of time and the rhythm of life on Earth. Ancient civilizations built monuments and aligned cities to track this east-west movement, creating calendars and predicting seasons. The east, associated with new beginnings, contrasts with the west, often symbolizing endings and reflection.',
        'North, south, east, and west form the foundation of our spatial understanding. These four cardinal directions enable us to create maps, navigate vast oceans, and explore unknown territories. They\'re deeply ingrained in human culture, appearing in everything from ancient spiritual practices to modern GPS systems. Each direction carries its own symbolism and significance across different societies, influencing architecture, agriculture, and even burial practices.',
        'South to north, west to east - these vectors of movement have shaped human migration and cultural exchange throughout history. From the Silk Road connecting East and West to the trans-Saharan trade routes linking North and South Africa, these pathways have facilitated the spread of ideas, technologies, and beliefs. In modern times, these directions continue to influence global trade winds, ocean currents, and even the flow of digital information across submarine cables.',
        'From the northwest to the southeast, diagonal paths cut across landscapes and cultural boundaries. This particular orientation often follows mountain ranges, river valleys, or ancient trade routes. In North America, the Northwest Passage sought by explorers contrasts with the Southeast\'s subtropical climate. Globally, this diagonal can represent a journey from cooler, more temperate regions to warmer, more tropical zones, each with its distinct ecosystems and human adaptations.',
        'Traveling eastward towards the rising sun has long symbolized hope, new beginnings, and discovery. This concept is deeply rooted in many cultures, from the American dream of westward expansion to the ancient Egyptian belief in the sun god Ra\'s daily rebirth in the east. In literature and film, journeys to the east often represent spiritual quests or the search for wisdom, while in practical terms, eastward travel challenges our circadian rhythms, leading to the phenomenon of jet lag.',
        'The compass points due north, an unwavering guide for travelers and explorers. This reliable orientation has been crucial in navigation, from ancient mariners using the North Star to modern GPS systems calculating position relative to the Earth\'s magnetic poles. The concept of "true north" extends beyond geography into philosophy and personal development, representing unwavering principles or goals. In many cultures, the north is associated with wisdom, introspection, and the challenges of harsh environments.',
        'Westward expansion across the continent has been a recurring theme in human history, from the ancient Greeks colonizing the western Mediterranean to the American frontier movement. This drive towards the setting sun often symbolizes adventure, opportunity, and the unknown. It has shaped national identities, inspired literature and art, and led to significant cultural exchanges and conflicts. The west, in many traditions, represents the end of the day\'s journey, a place of reflection and, sometimes, the afterlife.',
        'Southern winds bring warm air, influencing weather patterns, agricultural practices, and cultural traditions worldwide. In the northern hemisphere, these winds often herald the arrival of spring and summer, bringing relief from winter\'s chill. They play crucial roles in monsoon systems, affecting the livelihoods of billions. In sailing and aviation, understanding these wind patterns is essential for efficient travel. Culturally, the south is often associated with warmth, leisure, and a more relaxed pace of life in many societies.',
        'Northeast to southwest diagonal creates interesting geographical and cultural gradients across continents. In North America, this line might connect the bustling urban northeast with the arid southwest, spanning diverse ecosystems and ways of life. In Europe, it could link the fjords of Scandinavia with the Mediterranean coast. This diagonal often represents transitions in climate, vegetation, and human activity, from densely populated industrial regions to more sparsely inhabited natural landscapes.',
        'The ship sailed south by southeast, a course that might take it from temperate waters to tropical seas. This nautical bearing combines two directions, refining navigation for more precise journeys. Historically, such routes have been vital for trade, exploration, and cultural exchange, linking continents across vast oceans. In literature and film, journeys in this direction often symbolize adventure, escape, or the search for paradise. The southeast, in many cultures, is associated with growth, expansion, and new opportunities.',
        'Turn left to head west, right to go east - these simple instructions encapsulate the fundamentals of navigation and spatial awareness. The ability to orient oneself and make directional decisions is a critical cognitive skill, developed early in childhood and refined throughout life. This east-west axis often defines major roads and city layouts, influencing urban planning and daily commutes. In many cultures, east and west carry symbolic meanings: east associated with beginnings and enlightenment, west with endings and introspection.',
        'Northern lights dance in the arctic sky, a spectacular display of nature\'s beauty and the Earth\'s magnetic field interactions with solar winds. This phenomenon, also known as aurora borealis, has inspired myths, scientific inquiry, and tourism in northern regions. The north, often associated with cold and challenge, also represents resilience and adaptation in human cultures. From the Inuit in North America to the Sami in Europe, northern peoples have developed unique ways of life in harmony with their harsh environments.',
        'Easterly breeze off the ocean brings moisture and moderates coastal climates. This wind pattern plays a crucial role in weather systems, influencing precipitation, temperature, and air quality in coastal regions. In many cultures, the east is associated with new beginnings, enlightenment, and spiritual awakening, partly due to its connection with the rising sun. Easterly winds have been important in navigation, particularly during the age of sail, enabling trade routes and explorations that shaped global history.',
        'Southwest desert meets northwest forest, creating a dramatic ecological transition zone. This contrast showcases the diversity of landscapes and ecosystems within a single region or country. The southwest, often characterized by arid climates and unique adaptations of flora and fauna, contrasts sharply with the lush, temperate forests of the northwest. These geographical differences have shaped human settlements, economic activities, and cultural practices, leading to diverse regional identities within larger national frameworks.',
        'From sea to shining sea, east to west, this phrase often evokes images of vast continental expanses and national unity. It\'s particularly associated with the United States, spanning from the Atlantic to the Pacific. This east-west orientation has shaped nations\' development, from ancient China\'s unification to Russia\'s expansion across Eurasia. It influences climate patterns, time zones, and cultural variations within countries. The journey from east to west is often symbolic of progress, adventure, and the fulfillment of manifest destiny in many national narratives.',
        'North Star guides travelers at night, a constant celestial beacon that has aided navigation for thousands of years. Also known as Polaris, this star appears stationary in the night sky due to its alignment with Earth\'s rotational axis. Its reliability has made it a powerful symbol in many cultures, representing constancy, guidance, and hope. In the northern hemisphere, the ability to locate the North Star has been a crucial skill for travelers, sailors, and escaped slaves seeking freedom.',
        'Southward migration of birds in winter is one of nature\'s most impressive phenomena. This annual journey, triggered by changing daylight and food availability, sees millions of birds traveling vast distances. The south, representing warmth and abundance in this context, becomes a seasonal refuge. This migratory pattern has influenced human cultures, from ancient augury practices to modern conservation efforts. It also serves as a poignant reminder of the interconnectedness of global ecosystems and the impacts of climate change.',
        'West coast to east coast road trip captures the essence of cross-country adventure in many nations. In the United States, this journey spans diverse landscapes, from the Pacific beaches through mountains and plains to the Atlantic seaboard. Such trips often symbolize personal growth, cultural exploration, and the discovery of national identity. The contrasts between west and east coasts - in climate, culture, and pace of life - offer travelers a comprehensive experience of their country\'s diversity.',
        'The Great Wall stretches east to west, a monumental testament to human engineering and determination. This ancient fortification, spanning thousands of kilometers across northern China, was built to protect against nomadic invasions. Its east-west orientation follows the natural boundaries between agricultural China and the steppes of Central Asia. The Wall has become a powerful symbol of Chinese civilization and the country\'s historical focus on protecting its heartland from northern threats.',
        'Northernmost point of the continent often represents the ultimate challenge for explorers and adventurers. Whether it\'s Cape Morris Jesup in Greenland or Point Barrow in Alaska, these extreme locations embody the human drive to push boundaries. The north, associated with cold, darkness, and isolation, has long fascinated scientists, explorers, and writers. Indigenous peoples of these regions have developed unique cultures adapted to these harsh environments, offering valuable lessons in resilience and sustainable living.',
        'Southeast Asian tropical climate shapes a region rich in biodiversity and cultural diversity. This part of the world, where the southeast orientation often leads to warm waters and lush landscapes, is home to ancient civilizations and modern economic powerhouses. The tropical monsoon climate influences agriculture, architecture, and daily life. Southeast Asia\'s location at the crossroads of major cultural influences - Indian, Chinese, and Western - has resulted in a fascinating blend of traditions and innovations.',
        'Western frontier of the old world once represented the edge of the known to ancient Mediterranean civilizations. This concept of a western limit shifted over time, from the Pillars of Hercules (Strait of Gibraltar) to the Americas. The west has often symbolized mystery, opportunity, and the unknown in various cultures. Historically, westward exploration led to significant cultural exchanges, conflicts, and the reshaping of global power dynamics, especially during the Age of Exploration and colonial periods.',
        'Due south to reach the equator, a journey that traverses climate zones and ecosystems. This meridional travel showcases the Earth\'s diverse environments, from temperate or polar regions to the tropics. The equator, an imaginary line dividing the planet into northern and southern hemispheres, plays a crucial role in climate patterns, daylight duration, and even satellite orbits. Cultures near the equator have developed unique adaptations to their environment, influencing everything from architecture to agricultural practices.',
        'East meets West at the international date line, an imaginary line on the surface of the Earth that runs from the North Pole to the South Pole and demarcates the change of one calendar day to the next. This concept highlights the arbitrary nature of timekeeping and date designation in a round world. The phrase "East meets West" also symbolizes cultural exchange and fusion, representing the intersection of Eastern and Western philosophies, traditions, and ways of life in our increasingly interconnected global society.'
    ]).test()
