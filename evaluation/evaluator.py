class Evaluator:
    def __init__(self):
        pass

    def evaluate_token_level(self, ground_truth, predictions):
        # Dummy token-level evaluation logic
        return {"precision": 0.9, "recall": 0.9, "f1": 0.9}

    def evaluate_region_detection(self, ground_truth, predictions):
        # Dummy region detection evaluation logic
        return {"mAP": 0.9}

    def evaluate_table_reconstruction(self, ground_truth, predictions):
        # Dummy table reconstruction evaluation logic
        return {"accuracy": 0.9}

    def evaluate(self, ground_truth, predictions):
        token_level_metrics = self.evaluate_token_level(ground_truth, predictions)
        region_detection_metrics = self.evaluate_region_detection(ground_truth, predictions)
        table_reconstruction_metrics = self.evaluate_table_reconstruction(ground_truth, predictions)

        return {
            "token_level": token_level_metrics,
            "region_detection": region_detection_metrics,
            "table_reconstruction": table_reconstruction_metrics
        }