import sys
from modules.data.data_preprocessing import preprocess_folder

'''
    Preprocess a given folder
    Args:
        folder_path: str
'''
if __name__ == '__main__':

    if len(sys.argv) != 3:
        print('Usage: python3 preprocess_main.py <source_folder_path> <target_folder_path>')
        sys.exit(1)

    source_folder_path, target_folder_path = sys.argv[1], sys.argv[2]

    preprocess_folder(source_folder_path, (80, 80), target_folder_path)