from modules.data.data_generation import generate_data
from modules.data.data_preprocessing import preprocess_folder
from modules.model.model import build_model, test_model

if __name__ == "__main__":  
    #preprocess_folder("data/train/neg", "data_pp/train/neg", (80, 80))
    #build_model("mini_data_pp/train/pos", "mini_data_pp/train/neg", "model.h5")
    pos_points, pos_accuracy, neg_points, neg_accuracy = test_model("model.h5", 0.5, "mini_data_pp/test/pos", "mini_data_pp/test/neg")
    print(f"Positive test points: {pos_points}, Positive accuracy: {pos_accuracy}")
    print(f"Negative test points: {neg_points}, Negative accuracy: {neg_accuracy}")
   