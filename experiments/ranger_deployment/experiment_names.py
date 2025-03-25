import random
import datetime

class FancyNameGenerator:
    
    def __init__(self):
        self.mythical = ['phoenix', 'dragon', 'sphinx', 'hydra', 'kraken', 'atlas', 'nova']
        self.cosmic = ['nebula', 'quasar', 'pulsar', 'vortex', 'horizon', 'nexus']
        self.elements = ['photon', 'quantum', 'neutron', 'plasma', 'vector', 'matrix']
        self.colors = ['crimson', 'azure', 'violet', 'amber', 'cobalt', 'emerald']
        
    def generate_name(self, style='random'):
        timestamp = datetime.datetime.now().strftime("%m-%d-%y__%H-%M")
        
        if style == 'mythical':
            base = random.choice(self.mythical)
        elif style == 'cosmic':
            base = random.choice(self.cosmic)
        elif style == 'quantum':
            base = random.choice(self.elements)
        elif style == 'chromatic':
            base = random.choice(self.colors)
        else:
            categories = [self.mythical, self.cosmic, self.elements, self.colors]
            words = [random.choice(category) for category in random.sample(categories, 1)]
            base = ''.join(words)
            
        return f"{base}_{timestamp}"


if __name__ == "__main__":

    generator = FancyNameGenerator()
    
    print("Random style:", generator.generate_name(), "\n")
    print("Mythical style:", generator.generate_name('mythical'), "\n")
    print("Cosmic style:", generator.generate_name('cosmic'), "\n")
    print("Quantum style:", generator.generate_name('quantum'), "\n")
    print("Chromatic style:", generator.generate_name('chromatic'), "\n")