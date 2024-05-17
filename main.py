from modules.data.data_generation import generate_data
from modules.data.data_preprocessing import preprocess_folder
from modules.model.model import build_model, test_model
from modules.util.print import print_model_performance

if __name__ == "__main__":  
    '''
    preprocess_folder("data/train/neg", "data_pp/train/neg", (80, 80))
    preprocess_folder("data/train/pos", "data_pp/train/pos", (80, 80))
    preprocess_folder("data/test/neg", "data_pp/test/neg", (80, 80))
    preprocess_folder("data/test/pos", "data_pp/test/pos", (80, 80))
    '''
    build_model("data_pp/train/pos", "data_pp/train/neg", "model.keras", 3, 15)
    pos_points, pos_accuracy, neg_points, neg_accuracy = test_model("model.keras", 0.5, "data_pp/test/pos", "data_pp/test/neg", True)
    print_model_performance(pos_points, pos_accuracy, neg_points, neg_accuracy)
   