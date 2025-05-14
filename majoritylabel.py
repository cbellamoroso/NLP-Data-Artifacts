from datasets import load_dataset
from collections import Counter


def process_dataset(dataset_name, split, label_key="label"):
    """
    Loads a Hugging Face dataset, filters out invalid labels (not 0, 1, or 2),
    and calculates label statistics.

    Args:
        dataset_name (str): The name of the Hugging Face dataset.
        split (str): The dataset split to load (e.g., 'validation').
        label_key (str): The key used for labels in the dataset.

    Returns:
        dict: Label statistics including counts and percentages.
    """
    # Load the dataset
    dataset = load_dataset(dataset_name, split=split)

    # Filter out invalid labels
    filtered_data = [entry for entry in dataset if entry[label_key] in {0, 1, 2}]

    # Count labels
    label_counts = Counter(entry[label_key] for entry in filtered_data)
    total_count = sum(label_counts.values())

    # Calculate percentages
    label_percentages = {label: (count / total_count) * 100 for label, count in label_counts.items()}

    return {
        "dataset_name": dataset_name,
        "split": split,
        "label_counts": dict(label_counts),
        "label_percentages": label_percentages,
    }


if __name__ == "__main__":
    # Datasets and splits
    datasets_to_process = [
        #{"name": "snli", "split": "validation"},
        {"name": "ANLI", "split": "test_r1"},
        {"name": "anli", "config": "r2", "split": "test_r2"},
        {"name": "anli", "config": "r3", "split": "test_r3"},
    ]

    # Process each dataset
    for dataset_info in datasets_to_process:
        name = dataset_info["name"]
        split = dataset_info["split"]
        config = dataset_info.get("config", None)

        if config:
            stats = process_dataset(f"{name}", split)
        else:
            stats = process_dataset(name, split)

        print(f"\nDataset: {stats['dataset_name']}, Split: {stats['split']}")
        print("Label Counts:", stats["label_counts"])
        print("Label Percentages:", stats["label_percentages"])
