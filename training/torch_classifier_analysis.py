import torch
import torchvision
from typing import List
from enum import Enum

import config
from torch_classification import build_embedding_network
import os
from torch_datasets import AudioDeterministicBirdDataset, TargetEncoding, read_species_list
import csv
from torch.utils.data import DataLoader
from tqdm import tqdm
from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer
import json
import sys

ANALYSIS_SUBDIR = "analysis"

MIN_BIRDNET_CONFIDENCE = 0.5
# ambiguous labels from xeno-canto and birdnet species list
BIRDNET_TRANSLATIONS = {
    "Curruca communis": "Sylvia communis"
}

def load_classifier_network(
        model_dir: str,
):
    model_path = os.path.join(model_dir, "siamese_network.pt")

    kwargs, model_state_dict = torch.load(model_path)

    model = build_embedding_network(
        **kwargs
    )

    model.load_state_dict(model_state_dict)
    model = model.to('cuda')

    return model


# from https://github.com/python/cpython/issues/60009
class RedirectStdout:
    ''' Create a context manager for redirecting sys.stdout
        to another file.
    '''

    def __init__(self, new_target):
        self.new_target = new_target

    def __enter__(self):
        self.old_target = sys.stdout
        sys.stdout = self.new_target
        return self

    def __exit__(self, exctype, excinst, exctb):
        sys.stdout = self.old_target


def analyze_classifier_test_set(
        model_dir: str,
        data_dir: str,
        split: str,
        species_list: List[str],
        min_sns: float = 50,
        out_fn: str = "classifier_outputs_test_all.csv",
        include_noise_label: bool = False
):
    classifier = load_classifier_network(model_dir)

    classifier.eval()

    dataset = AudioDeterministicBirdDataset(
        data_dir,
        split=split,
        min_sns=min_sns,
        species_list=species_list,
        target_encoding=TargetEncoding.ONE_HOT,
        mel_transforms=torch.nn.Sequential(
            torchvision.transforms.Resize(
                (128, 384),
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR
            ),
            torchvision.transforms.Normalize(-30.0118, 12.8596)
        )
    )

    print("Total Samples in Test Set: ", len(dataset))

    if len(dataset) == 0:
        return

    dataloader = DataLoader(dataset, batch_size=1024, num_workers=os.cpu_count())

    with torch.no_grad():
        count = 0

        out_dir = os.path.join(model_dir, ANALYSIS_SUBDIR)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        out_path = os.path.join(out_dir, out_fn)

        if os.path.isfile(out_path):
            os.remove(out_path)

        with open(out_path, "w") as f:
            csv_writer = csv.DictWriter(f, fieldnames=["i", "pred", "label"])
            csv_writer.writeheader()

            for batch_idx, (images, labels) in enumerate(tqdm(dataloader)):
                images = images.to('cuda')
                predictions = classifier(images)
                if include_noise_label:
                    predictions = predictions.round()
                else:
                    predictions = predictions.argmax(dim=1)

                rows = []

                for num in range(len(images)):
                    if include_noise_label:
                        pred_label = predictions[num]
                        indices = (pred_label == 1.0).nonzero()
                        assert (len(indices) == 0 or len(indices) == 1)
                        if len(indices) == 0:
                            pred_label = 0
                        else:
                            pred_label = indices[0] + 1

                        label = labels[num].argmax() + 1
                    else:
                        # no add 1
                        pred_label = predictions[num]
                        label = labels[num].argmax()

                    rows.append({
                        "i": count,
                        "pred": int(pred_label),
                        "label": int(label)
                    })
                    count += 1

                csv_writer.writerows(rows)


def analyze_birdnet_split(
        directory: str,
        species_list: str,
        split: str = "test",
        out_directory: str = "./test",
        min_sns=None
):
    if not os.path.exists(out_directory):
        os.makedirs(out_directory)

    confusion_metrics = {}

    species_list = read_species_list(species_list=species_list)

    analyzer = Analyzer()

    for species in species_list:
        confusion_metrics[species] = {
            "tp": 0,
            "fp": 0,
            "tn": 0,
            "fn": 0
        }

    file_paths = []

    for species in species_list:
        split_path = os.path.join(directory, species, split)
        species_file_paths = os.listdir(split_path)

        assert (len(species_file_paths) != 0)

        for file in species_file_paths:
            if min_sns is not None:
                sns = float(file.split("_")[1])
                if sns > min_sns:
                    file_paths.append(os.path.join(split_path, file))

    rows = []

    count = 0
    for file_path in tqdm(file_paths):
        species = file_path.split("/")[-3]

        recording = Recording(
            analyzer,
            file_path,
            min_conf=MIN_BIRDNET_CONFIDENCE
        )

        with RedirectStdout(None):
            recording.analyze()

        if len(recording.detections) != 0:
            detection = max(recording.detections, key=lambda x: x["confidence"])
        else:
            # need to count one fn for 'species'
            confusion_metrics[species]["fn"] += 1
            rows.append({
                "i": count,
                "pred": 0,
                "label": int(species_list.index(species)) + 1
            })
            count += 1
            continue

        pred_species = detection["scientific_name"]

        if pred_species in BIRDNET_TRANSLATIONS.keys():
            pred_species = BIRDNET_TRANSLATIONS[pred_species]

        # 0 is the background / unknown vocalization class
        rows.append({
            "i": count,
            "pred": int(species_list.index(pred_species) + 1) if pred_species in species_list else 0,
            "label": int(species_list.index(species)) + 1
        })
        count += 1

        if species == pred_species:
            confusion_metrics[species]["tp"] += 1
            for other_species in species_list:
                if other_species != pred_species:
                    confusion_metrics[other_species]["tn"] += 1
        if species != pred_species:
            confusion_metrics[species]["fn"] += 1
            if pred_species in species_list:
                confusion_metrics[pred_species]["fp"] += 1
            for other_species in species_list:
                if other_species != species and other_species != pred_species:
                    confusion_metrics[other_species]["tn"] += 1

    total_tp = 0
    for metrics in confusion_metrics.values():
        total_tp += metrics["tp"]

    print("Count: ", count)
    print("Classifier Accuracy: ", total_tp / count)


    precisions = {
        sp: 0 if (confusion_matrix["tp"] + confusion_matrix["fp"] == 0) else confusion_matrix["tp"] / (confusion_matrix["tp"] + confusion_matrix["fp"])
        for (sp, confusion_matrix) in confusion_metrics.items()
    }

    recalls = {
        sp: confusion_matrix["tp"] / (confusion_matrix["tp"] + confusion_matrix["fn"])
        for (sp, confusion_matrix) in confusion_metrics.items()
    }

    supports = {
        sp: confusion_matrix["tp"] + confusion_matrix["fn"]
        for (sp, confusion_matrix) in confusion_metrics.items()
    }

    averaged_metrics = {
        "classes": {
            sp: {
                "precision": confusion_matrix["tp"] / (confusion_matrix["tp"] + confusion_matrix["fp"]) if (confusion_matrix["tp"] + confusion_matrix["fp"]) != 0 else 1,
                "recall": confusion_matrix["tp"] / (confusion_matrix["tp"] + confusion_matrix["fn"]),
                "support": confusion_matrix["tp"] + confusion_matrix["fn"]
            } for (sp, confusion_matrix) in confusion_metrics.items()
        }
    }

    for sp in species_list:
        precision = precisions[sp]
        recall = recalls[sp]
        support = supports[sp]

        print(f"{sp} - Prec: {precision} - Recall: {recall} - Support: {support}")

    confusion_matrix_fn = os.path.join(out_directory, "confusion_matrix.json")

    with open(confusion_matrix_fn, "w") as f:
        json.dump(confusion_metrics, f, indent=4)

    macro_precision = 0
    macro_recall = 0

    for species in species_list:
        macro_precision += averaged_metrics["classes"][species]["precision"]
        macro_recall += averaged_metrics["classes"][species]["recall"]

    macro_precision /= len(species_list)
    macro_recall /= len(species_list)

    print("Macro Precision: ", macro_precision)
    print("Macro Recall: ", macro_recall)

    tp = sum([confusion_metrics[sp]["tp"] for sp in species_list])
    # tn = sum([confusion_metrics[sp]["tn"] for sp in species_list])
    fp = sum([confusion_metrics[sp]["fp"] for sp in species_list])
    fn = sum([confusion_metrics[sp]["fn"] for sp in species_list])

    micro_precision = tp / (tp + fp)
    micro_recall = tp / (tp + fn)

    print("Micro Precision: ", micro_precision)
    print("Micro Recall: ", micro_recall)

    weighted_precision = 0
    weighted_recall = 0

    total_support = 0
    for sp in species_list:
        support = supports[sp]
        weighted_precision += averaged_metrics["classes"][sp]["precision"] * support
        weighted_recall += averaged_metrics["classes"][sp]["recall"] * support
        total_support += support
    weighted_precision /= total_support
    weighted_recall /= total_support
    print("Weighted Precision: ", weighted_precision)
    print("Weighted Recall: ", weighted_recall)

    averaged_metrics["averages"] = {}

    averaged_metrics["averages"]["macro"] = {
        "precision": macro_precision,
        "recall": macro_recall
    }
    averaged_metrics["averages"]["micro"] = {
        "precision": micro_precision,
        "recall": micro_recall
    }
    averaged_metrics["averages"]["weighted"] = {
        "precision": weighted_precision,
        "recall": weighted_recall
    }


    average_metrics_fn = os.path.join(out_directory, "metrics.json")

    with open(average_metrics_fn, "w") as f:
        json.dump(averaged_metrics, f, indent=4)

    classifications_fn = os.path.join(out_directory, "classification_output.csv")

    with open(classifications_fn, "w") as f:
        writer = csv.DictWriter(f, fieldnames=["i", "pred", "label"])
        writer.writeheader()
        writer.writerows(rows)



if __name__ == "__main__":
    models_dir = os.path.join(config.MODELS_DIR, "classification", "model-3600")

    data_dir = "./data/split-3600"

    unseen_data_dir = "./data/unseen"

    esc_dir = config.ESC_DIR

    species_list = read_species_list("./species-3600.txt")

    analyze_classifier_test_set(
        model_dir=models_dir,
        min_sns=50,
        data_dir=data_dir,
        species_list=species_list,
        out_fn="classifier_outputs_test_all.csv",
        include_noise_label=True,
        split="test"
    )

    analyze_birdnet_split(
        directory="./split-3600/split-song-A/audio/",
        split="test",
        out_directory="./birdnet-3600",
        species_list="./species-3600.txt",
        min_sns=50
    )
