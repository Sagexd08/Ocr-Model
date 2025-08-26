import argparse

def evaluate(dataset):
    # Dummy evaluation logic
    print(f"Evaluating on dataset: {dataset}")
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Name of the dataset to evaluate on")
    args = parser.parse_args()

    evaluate(args.dataset)