import os
from typing import Optional
from .imagelist import ImageList

class DomainNet(ImageList):
    """`DomainNet <http://ai.bu.edu/M3SDA/>`_ Dataset.

    Args:
        root (str): Root directory of dataset
        task (str): The task (domain) to create dataset. Choices include ``'clipart'``, \
            ``'infograph'``, ``'painting'``, ``'quickdraw'``, ``'real'`` and ``'sketch'``.
        download (bool, optional): If true, downloads the dataset from the internet and puts it \
            in root directory. If dataset is already downloaded, it is not downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image and returns a \
            transformed version. E.g, :class:`torchvision.transforms.RandomCrop`.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.

    .. note:: In `root`, there will exist following files after downloading.
        ::
            clipart/
                *.jpg
            infograph/
            painting/
            quickdraw/
            real/
            sketch/
            image_list/
                clipart.txt
                infograph.txt
                painting.txt
                quickdraw.txt
                real.txt
                sketch.txt
    """
    # image_list = {
    #     "clipart": "_image_list/clipart_list.txt",
    #     "infograph": "_image_list/infograph_list.txt",
    #     "painting": "_image_list/painting_list.txt",
    #     "quickdraw": "_image_list/quickdraw_list.txt",
    #     "real": "_image_list/real_list.txt",
    #     "sketch": "_image_list/sketch_list.txt",
    # }

    CLASSES = ['pencil', 'The_Mona_Lisa', 'cookie', 'arm', 'panda', 
    'baseball_bat', 'chandelier', 'fence', 'frying_pan', 'cloud', 'pig', 
    'book', 'mermaid', 'foot', 'stove', 'power_outlet', 'diamond', 'scissors', 
    'key', 'hospital', 'pond', 'car', 'trumpet', 'bracelet', 'lipstick', 'flying_saucer', 
    'ear', 'wheel', 'cannon', 'binoculars', 'laptop', 'ant', 'hockey_stick', 'bed', 
    'cell_phone', 'bread', 'sailboat', 'sweater', 'hat', 'cat', 'flamingo', 'drums', 
    'crown', 'lightning', 'animal_migration', 'pizza', 'roller_coaster', 'elbow', 
    'umbrella', 'finger', 'sheep', 'duck', 'bridge', 'mouth', 'hot_tub', 'washing_machine', 
    'wristwatch', 'backpack', 'popsicle', 'skateboard', 'mushroom', 'The_Eiffel_Tower', 
    'matches', 'string_bean', 'tractor', 'squirrel', 'bottlecap', 'pineapple', 'hedgehog', 
    'firetruck', 'saxophone', 'stethoscope', 'fire_hydrant', 'raccoon', 'shark', 'donut', 
    'flower', 'blueberry', 'guitar', 'rabbit', 'bowtie', 'axe', 'bee', 'basket', 'boomerang', 
    'pants', 'fireplace', 'spreadsheet', 'snorkel', 'ocean', 'bear', 'soccer_ball', 'rake', 
    'dragon', 'toothpaste', 'paint_can', 'rollerskates', 'paintbrush', 'rainbow', 'triangle', 
    'map', 'bathtub', 'ladder', 'church', 'cake', 'tennis_racquet', 'helmet', 'alarm_clock', 
    'necklace', 'chair', 'stop_sign', 'owl', 'golf_club', 'saw', 'stitches', 'snowman', 
    'knife', 'swing_set', 'campfire', 'skyscraper', 'rifle', 'police_car', 'stereo', 
    'calculator', 'butterfly', 'crab', 'piano', 'palm_tree', 'teddy-bear', 'ceiling_fan', 
    'cooler', 'anvil', 'waterslide', 'potato', 'picture_frame', 'passport', 'windmill', 
    'yoga', 'sleeping_bag', 'beard', 'canoe', 'envelope', 'harp', 'giraffe', 'kangaroo', 
    'camera', 'shoe', 'moon', 'see_saw', 'mailbox', 'shovel', 'banana', 'square', 
    'remote_control', 'fish', 'crayon', 'screwdriver', 'blackberry', 'cactus', 'mountain', 
    'lobster', 'zebra', 'squiggle', 'cow', 'monkey', 'clock', 'hurricane', 'spider', 
    'octagon', 'violin', 'elephant', 'toe', 'tree', 'pillow', 'house_plant', 
    'hockey_puck', 'diving_board', 'sea_turtle', 'star', 'ambulance', 'traffic_light', 
    'toaster', 'van', 'floor_lamp', 'nail', 'broom', 'smiley_face', 'garden', 
    'strawberry', 'face', 'broccoli', 'wine_glass', 'goatee', 'crocodile', 'eyeglasses', 
    'coffee_cup', 'lollipop', 'postcard', 'onion', 'calendar', 'megaphone', 'drill', 
    'The_Great_Wall_of_China', 'clarinet', 'camel', 'castle', 'sock', 'trombone', 
    'streetlight', 'hourglass', 'sink', 't-shirt', 'computer', 'zigzag', 'cup', 
    'camouflage', 'oven', 'feather', 'sandwich', 'octopus', 'dishwasher', 'sword', 
    'beach', 'bus', 'bat', 'pool', 'mouse', 'lighter', 'hammer', 'line', 'carrot', 
    'river', 'lighthouse', 'eraser', 'wine_bottle', 'keyboard', 'motorbike', 'paper_clip', 
    'snake', 'candle', 'sun', 'table', 'stairs', 'tent', 'toothbrush', 'penguin', 'toilet', 
    'fork', 'horse', 'pliers', 'headphones', 'flip_flops', 'peanut', 'birthday_cake', 
    'hot_air_balloon', 'rain', 'dolphin', 'mosquito', 'shorts', 'vase', 'bicycle', 'whale', 
    'flashlight', 'watermelon', 'microphone', 'cello', 'brain', 'tornado', 'jail', 'swan', 
    'lantern', 'scorpion', 'tiger', 'spoon', 'snail', 'aircraft_carrier', 'compass', 'underwear', 
    'parachute', 'pickup_truck', 'airplane', 'bucket', 'couch', 'garden_hose', 'cruise_ship', 
    'suitcase', 'ice_cream', 'radio', 'belt', 'bird', 'door', 'leg', 'house', 'purse', 'eye', 
    'grapes', 'basketball', 'television', 'nose', 'pear', 'dumbbell', 'hamburger', 'dresser', 
    'steak', 'moustache', 'syringe', 'truck', 'bench', 'baseball', 'parrot', 'telephone', 
    'apple', 'marker', 'bulldozer', 'light_bulb', 'skull', 'microwave', 'leaf', 'snowflake', 
    'fan', 'hexagon', 'peas', 'school_bus', 'circle', 'teapot', 'jacket', 'bandage', 'asparagus', 
    'helicopter', 'knee', 'train', 'hand', 'dog', 'barn', 'lion', 'bush', 'frog', 'speedboat', 
    'angel', 'tooth', 'mug', 'rhinoceros', 'submarine', 'hot_dog', 'grass']

    def __init__(self, root: str, task: str, download: Optional[bool] = False, **kwargs):
        # assert task in self.image_list
        # data_list_file = os.path.join(root, self.image_list[task])
        
        data_list_file = os.path.join(root, f"_image_list/{task}_list.txt")

        super(DomainNet, self).__init__("/data", DomainNet.CLASSES, data_list_file=data_list_file, **kwargs)
    @classmethod
    def domains(cls):
        return list(cls.image_list.keys())  