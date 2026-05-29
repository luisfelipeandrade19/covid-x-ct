from dataset import ctxcovid
import os

class Config:
    NUM_CLASSES = 3
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    EPOCHS_PER_STAGE = 5       
    MAX_UNFREEZE_STAGE = 4
    MAX_EPOCHS = 25
    BASE_PATH = ctxcovid
    IMAGES_DIR = os.path.join(BASE_PATH, '3A_images')