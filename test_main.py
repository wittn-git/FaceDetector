from modules.model.model import test_model
from modules.util.print import print_model_performance
import sys

'''
    Test the model
    Args:
        model_path: str
        threshold: float
        pos_folder: str
        neg_folder: str
'''
if __name__ == "__main__":

    if len(sys.argv) != 6:
        print('Usage: python3 test_main.py <model_path> <threshold> <pos_folder> <neg_folder> <data_preprocessed>')
        sys.exit(1)
    
    model_path, threshold, pos_folder, neg_folder, data_preprocessed = sys.argv[1], float(sys.argv[2]), sys.argv[3], sys.argv[4], bool(sys.argv[5])

    pos_points, pos_accuracy, neg_points, neg_accuracy, time_per_sample = test_model(model_path, threshold, pos_folder, neg_folder, data_preprocessed)
    print_model_performance(pos_points, pos_accuracy, neg_points, neg_accuracy, time_per_sample)