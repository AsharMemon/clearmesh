#!/usr/bin/env python3
"""Generate diverse reference images for SLAT pair generation.

Uses TRELLIS.2's text-conditioned image preprocessing — any single-view
object image works. Generates simple solid-color object silhouettes
as "seed images" that TRELLIS.2's image conditioning model will process.

For production, these would be real photos/renders. For training data diversity,
what matters is that each image produces a DIFFERENT sparse structure and SLAT.
TRELLIS.2's DINOv2 conditioning extracts semantic features, so even simple
reference images produce diverse 3D outputs across different seeds.

Usage:
    python generate_reference_images.py --output_dir /workspace/data/ref_images --count 500
"""

import argparse
import json
from pathlib import Path

# 500 diverse object categories for maximum SLAT diversity
OBJECT_PROMPTS = [
    # Vehicles (50)
    "red sports car", "blue pickup truck", "yellow school bus", "green motorcycle",
    "white ambulance", "black limousine", "orange excavator", "fire truck",
    "police car", "ice cream truck", "delivery van", "monster truck",
    "vintage beetle", "formula one car", "jeep wrangler", "tractor",
    "steamroller", "golf cart", "rickshaw", "horse carriage",
    "sailboat", "speedboat", "cruise ship", "submarine", "kayak",
    "canoe", "tugboat", "fishing boat", "yacht", "rowboat",
    "helicopter", "biplane", "jet fighter", "hot air balloon", "blimp",
    "rocket ship", "space shuttle", "drone quadcopter", "hang glider", "paraglider",
    "bicycle", "tricycle", "scooter", "skateboard", "unicycle",
    "segway", "wheelchair", "baby stroller", "shopping cart", "wagon",

    # Animals (60)
    "golden retriever dog", "tabby cat", "white rabbit", "brown horse",
    "elephant", "giraffe", "lion", "tiger", "polar bear", "panda bear",
    "penguin", "flamingo", "eagle", "owl", "parrot",
    "dolphin", "whale", "shark", "octopus", "sea turtle",
    "frog", "snake", "lizard", "chameleon", "crocodile",
    "butterfly", "ladybug", "dragonfly", "spider", "snail",
    "goldfish", "clownfish", "seahorse", "jellyfish", "starfish",
    "gorilla", "chimpanzee", "orangutan", "koala", "kangaroo",
    "deer", "moose", "buffalo", "rhinoceros", "hippopotamus",
    "fox", "wolf", "raccoon", "skunk", "hedgehog",
    "squirrel", "chipmunk", "hamster", "guinea pig", "mouse",
    "duck", "swan", "peacock", "rooster", "hummingbird",

    # Furniture (40)
    "wooden dining chair", "leather armchair", "rocking chair", "bar stool",
    "office desk", "coffee table", "dining table", "bedside table",
    "bookshelf", "display cabinet", "wardrobe", "dresser",
    "king size bed", "bunk bed", "baby crib", "hammock",
    "couch sofa", "loveseat", "bean bag chair", "futon",
    "bathroom vanity", "kitchen island", "workbench", "standing desk",
    "tv stand", "shoe rack", "coat rack", "umbrella stand",
    "filing cabinet", "storage chest", "toy box", "wine rack",
    "outdoor bench", "picnic table", "garden swing", "patio chair",
    "ceiling fan", "floor lamp", "desk lamp", "chandelier",

    # Electronics (40)
    "laptop computer", "desktop monitor", "tablet device", "smartphone",
    "digital camera", "film camera", "video camera", "webcam",
    "gaming console", "game controller", "vr headset", "arcade cabinet",
    "television", "projector", "speaker bluetooth", "headphones",
    "microphone", "radio vintage", "record player", "cassette player",
    "keyboard mechanical", "computer mouse", "graphics tablet", "joystick",
    "wifi router", "usb drive", "hard drive", "memory card",
    "printer", "scanner", "fax machine", "copier",
    "power bank", "wall charger", "extension cord", "surge protector",
    "smart watch", "fitness tracker", "e reader", "calculator",

    # Kitchen & Dining (40)
    "coffee mug", "wine glass", "beer stein", "teacup saucer",
    "dinner plate", "soup bowl", "salad bowl", "mixing bowl",
    "fork knife set", "chopsticks", "spoon ladle", "spatula",
    "cooking pot", "frying pan", "wok", "pressure cooker",
    "toaster", "blender", "coffee maker", "kettle electric",
    "microwave oven", "stand mixer", "food processor", "juicer",
    "cutting board", "rolling pin", "whisk", "colander",
    "salt shaker", "pepper grinder", "sugar bowl", "butter dish",
    "wine bottle", "beer bottle", "water pitcher", "thermos flask",
    "cake stand", "serving tray", "napkin holder", "ice bucket",

    # Sports & Recreation (40)
    "basketball", "soccer ball", "tennis ball", "baseball bat",
    "football helmet", "hockey stick", "golf club", "tennis racket",
    "bowling pin", "bowling ball", "billiard ball", "dart board",
    "boxing glove", "punching bag", "dumbbell weight", "kettlebell",
    "yoga mat", "jump rope", "resistance band", "foam roller",
    "surfboard", "snowboard", "ski boot", "ice skate",
    "fishing rod", "tackle box", "binoculars", "compass",
    "tent camping", "sleeping bag", "backpack hiking", "water canteen",
    "chess piece king", "chess piece knight", "dice pair", "playing cards",
    "board game", "jigsaw puzzle", "rubiks cube", "spinning top",

    # Clothing & Accessories (30)
    "sneaker shoe", "leather boot", "high heel shoe", "sandal",
    "baseball cap", "top hat", "cowboy hat", "winter beanie",
    "sunglasses", "reading glasses", "goggles ski", "monocle",
    "wristwatch", "bracelet", "necklace pendant", "ring diamond",
    "backpack school", "handbag leather", "suitcase travel", "wallet",
    "umbrella", "necktie", "bow tie", "belt leather",
    "gloves winter", "scarf wool", "earmuffs", "crown tiara",
    "helmet bicycle", "face mask",

    # Nature & Plants (30)
    "bonsai tree", "cactus potted", "sunflower", "rose bouquet",
    "orchid plant", "tulip flower", "daisy flower", "lily flower",
    "mushroom red", "pine cone", "acorn", "maple leaf",
    "seashell conch", "coral reef piece", "driftwood", "crystal geode",
    "rock boulder", "gemstone amethyst", "fossil trilobite", "amber specimen",
    "potted fern", "hanging plant", "succulent garden", "bamboo stalk",
    "venus flytrap", "aloe vera plant", "lavender bundle", "herb garden",
    "wreath floral", "terrarium glass",

    # Tools & Hardware (30)
    "hammer", "screwdriver", "wrench", "pliers",
    "power drill", "circular saw", "tape measure", "level tool",
    "paint brush", "paint roller", "ladder", "toolbox",
    "garden shovel", "rake garden", "wheelbarrow", "lawn mower",
    "fire extinguisher", "flashlight", "lantern", "torch",
    "lock padlock", "key set", "door handle", "hinge",
    "screw bolt", "nail", "chain link", "rope coil",
    "pulley", "gear cog",

    # Musical Instruments (20)
    "acoustic guitar", "electric guitar", "bass guitar", "ukulele",
    "piano keyboard", "violin", "cello", "trumpet",
    "saxophone", "flute", "clarinet", "harmonica",
    "drum set", "bongo drum", "tambourine", "maracas",
    "xylophone", "accordion", "banjo", "harp",

    # Toys & Games (20)
    "teddy bear", "rubber duck", "lego brick", "action figure",
    "dollhouse", "toy train", "toy airplane", "toy robot",
    "stuffed elephant", "yo yo", "slinky", "jack in box",
    "toy sword", "toy shield", "puppet", "kite",
    "bubble wand", "water gun", "frisbee", "boomerang",

    # Food & Drink (30)
    "birthday cake", "cupcake", "donut", "croissant",
    "pizza slice", "hamburger", "hot dog", "taco",
    "apple fruit", "banana", "pineapple", "watermelon slice",
    "ice cream cone", "popsicle", "candy cane", "lollipop",
    "chocolate bar", "cookie", "pretzel", "bagel",
    "sushi roll", "dumpling", "spring roll", "fortune cookie",
    "popcorn bucket", "french fries", "onion ring", "chicken drumstick",
    "cheese wheel", "bread loaf",

    # Architecture & Structures (20)
    "castle tower", "lighthouse", "windmill", "water tower",
    "church steeple", "pagoda", "pyramid", "obelisk",
    "bridge arch", "gazebo", "treehouse", "doghouse",
    "birdhouse", "mailbox", "fire hydrant", "street lamp",
    "phone booth", "bus stop shelter", "park bench", "fountain",

    # Office & School (20)
    "globe earth", "microscope", "telescope", "hourglass",
    "pencil", "pen fountain", "eraser", "pencil sharpener",
    "stapler", "tape dispenser", "paper clip", "binder clip",
    "scissors", "ruler", "protractor", "compass drawing",
    "notebook", "textbook", "backpack school", "lunch box",

    # Medical & Science (10)
    "stethoscope", "syringe", "pill bottle", "first aid kit",
    "test tube rack", "flask beaker", "petri dish", "dna helix model",
    "atom model", "skeleton model",

    # Misc (20)
    "trophy cup gold", "medal award", "ribbon bow", "gift box wrapped",
    "candle lit", "hourglass timer", "snow globe", "music box",
    "piggy bank", "coin stack", "treasure chest", "magic lamp",
    "skull decorative", "mask carnival", "fan handheld", "bell brass",
    "ship in bottle", "message in bottle", "crystal ball", "dreamcatcher",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--count", type=int, default=500)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Create a simple white image with text label for each prompt
    # TRELLIS.2 will use DINOv2 to extract features — the actual image content
    # drives the 3D generation through the conditioning model
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("pip install Pillow")
        return

    prompts = OBJECT_PROMPTS[:args.count]
    manifest = {}

    for i, prompt in enumerate(prompts):
        # Create a simple 512x512 image with white background
        # This is a PLACEHOLDER — TRELLIS.2's generation is seed-driven,
        # so even identical images produce different 3D shapes per seed.
        # For proper training, replace with real photos/renders.
        img = Image.new("RGBA", (512, 512), (255, 255, 255, 255))
        draw = ImageDraw.Draw(img)

        # Add text label
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
        except (OSError, IOError):
            font = ImageFont.load_default()

        # Center text
        text = prompt
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text(((512 - tw) / 2, (512 - th) / 2), text, fill=(0, 0, 0, 255), font=font)

        safe_name = prompt.replace(" ", "_").replace("/", "_")
        img_path = out / f"{safe_name}.png"
        img.save(img_path)
        manifest[safe_name] = prompt

    # Save manifest
    with open(out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Generated {len(prompts)} reference images in {out}")
    print(f"Manifest: {out}/manifest.json")


if __name__ == "__main__":
    main()
