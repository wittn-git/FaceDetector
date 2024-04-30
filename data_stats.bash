data_folder="data"

for folder in train test val; do
    for subfolder in pos neg; do
        subfolder_path="$data_folder/$folder/$subfolder"
        file_count=$(ls -1 "$subfolder_path" | wc -l)
        echo "Folder: $subfolder_path, File count: $file_count"
    done
done