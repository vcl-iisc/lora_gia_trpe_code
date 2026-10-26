# CIFAR-10
cifar10_classes = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# CIFAR-100 (example subset)
cifar100_superclasses = [
    "aquatic mammals",
    "fish",
    "flowers",
    "food containers",
    "fruit and vegetables",
    "household electrical devices",
    "household furniture",
    "insects",
    "large carnivores",
    "large man-made outdoor things",
    "large natural outdoor scenes",
    "large omnivores and herbivores",
    "medium-sized mammals",
    "non-insect invertebrates",
    "people",
    "reptiles",
    "small mammals",
    "trees",
    "vehicles 1",
    "vehicles 2",
]

cifar100_classes = [
    # aquatic mammals
    "beaver", "dolphin", "otter", "seal", "whale",

    # fish
    "aquarium fish", "flatfish", "ray", "shark", "trout",

    # flowers
    "orchids", "poppies", "roses", "sunflowers", "tulips",

    # food containers
    "bottles", "bowls", "cans", "cups", "plates",

    # fruit and vegetables
    "apples", "mushrooms", "oranges", "pears", "sweet peppers",

    # household electrical devices
    "clock", "computer keyboard", "lamp", "telephone", "television",

    # household furniture
    "bed", "chair", "couch", "table", "wardrobe",

    # insects
    "bee", "beetle", "butterfly", "caterpillar", "cockroach",

    # large carnivores
    "bear", "leopard", "lion", "tiger", "wolf",

    # large man-made outdoor things
    "bridge", "castle", "house", "road", "skyscraper",

    # large natural outdoor scenes
    "cloud", "forest", "mountain", "plain", "sea",

    # large omnivores and herbivores
    "camel", "cattle", "chimpanzee", "elephant", "kangaroo",

    # medium-sized mammals
    "fox", "porcupine", "possum", "raccoon", "skunk",

    # non-insect invertebrates
    "crab", "lobster", "snail", "spider", "worm",

    # people
    "baby", "boy", "girl", "man", "woman",

    # reptiles
    "crocodile", "dinosaur", "lizard", "snake", "turtle",

    # small mammals
    "hamster", "mouse", "rabbit", "shrew", "squirrel",

    # trees
    "maple", "oak", "palm", "pine", "willow",

    # vehicles 1
    "bicycle", "bus", "motorcycle", "pickup truck", "train",

    # vehicles 2
    "lawn-mower", "rocket", "streetcar", "tank", "tractor",
]

imagenet_categories = [
    "arachnid",
    "armadillo",
    "bear",
    "bird",
    "bug",
    "butterfly",
    "cat",
    "coral",
    "crocodile",
    "crustacean",
    "dinosaur",
    "dog",
    "echinoderms",
    "ferret",
    "fish",
    "flower",
    "frog",
    "fruit",
    "fungus",
    "hog",
    "lizard",
    "marine mammals",
    "marsupial",
    "mollusk",
    "mongoose",
    "monotreme",
    "person",
    "plant",
    "primate",
    "rabbit",
    "rodent",
    "salamander",
    "shark",
    "sloth",
    "snake",
    "trilobite",
    "turtle",
    "ungulate",
    "vegetable",
    "wild cat",
    "wild dog",
    "accessory",
    "aircraft",
    "ball",
    "boat",
    "building",
    "clothing",
    "container",
    "cooking",
    "decor",
    "electronics",
    "fence",
    "food",
    "furniture",
    "hat",
    "instrument",
    "lab equipment",
    "other",
    "outdoor scene",
    "paper",
    "sports equipment",
    "technology",
    "tool",
    "toy",
    "train",
    "vehicle",
    "weapon",
]

from torchvision.models import ResNet50_Weights

weights = ResNet50_Weights.DEFAULT
imagenet_classes = weights.meta["categories"]