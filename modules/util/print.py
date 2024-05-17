def print_model_performance(pos_points : int, pos_accuracy : float, neg_points : int, neg_accuracy : float, time_per_sample : float):
    
    TP, FN, FP = pos_accuracy * pos_points, (1 - pos_accuracy) * pos_points, (1 - neg_accuracy) * neg_points

    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    
    print(f"Number of positive test points: {pos_points}")
    print(f"Number of negative test points: {neg_points}")
    print(f"Accuracy: {(pos_accuracy + neg_accuracy) / 2}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1_score}")
    print(f"Time per sample: {time_per_sample}s")