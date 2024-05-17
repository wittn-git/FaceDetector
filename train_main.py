from modules.model.model import build_model, test_model
import sys

'''
    Train the model
    Args:
        pos_folder: str
        neg_folder: str
        model_path: str
        epochs: int
        batch_size: int
        steps_per_epoch: int
'''
if __name__ == "__main__":

    if len(sys.argv) != 6:
        print('Usage: python3 train_main.py <pos_folder> <neg_folder> <model_path> <epochs> <batch_size> <steps_per_epoch>')
        sys.exit(1)
    
    pos_folder, neg_folder, model_path = sys.argv[1], sys.argv[2], sys.argv[3]
    epochs, batch_size, steps_per_epoch = int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])

    build_model(pos_folder, neg_folder, model_path, epochs, batch_size, steps_per_epoch)