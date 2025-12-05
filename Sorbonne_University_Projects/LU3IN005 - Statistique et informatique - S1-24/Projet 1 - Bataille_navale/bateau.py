# Sam ASLO 21210657
# Yuxiang ZHANG 21202829

class Bateau:
    bat_types = {
        "porte-avions": 1,
        "croiseur": 2,
        "contre-torpilleurs": 3,
        "sous-marin": 4,
        "torpilleur": 5,
    }
    bat_sizes = {
        "porte-avions": 5,
        "croiseur": 4,
        "contre-torpilleurs": 3,
        "sous-marin": 3,
        "torpilleur": 2,
    }
    bat_names = {v: k for k, v in bat_types.items()}

    def __init__(self, name):
        self.name = name
        self.num = Bateau.bat_types[name]
        self.size = Bateau.bat_sizes[name]
        self.direction = None
        self.position = None
        self.hit_positions = []

