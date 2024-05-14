from modules.data.data_generation import generate_data
from modules.data.data_preprocessing import preprocess_folder

if __name__ == "__main__":
    preprocess_folder("mini_data/train/pos", "mini_data_pp/train/pos", (80, 80))
    preprocess_folder("mini_data/train/neg", "mini_data_pp/train/neg", (80, 80))
    preprocess_folder("mini_data/test/pos", "mini_data_pp/test/pos", (80, 80))
    preprocess_folder("mini_data/test/neg", "mini_data_pp/test/neg", (80, 80))