import csv
import os
from typing import List, Union

import torch
import torch.nn.functional
import torchvision.transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from torch_datasets import TargetEncoding, AudioDeterministicBirdDataset, \
    read_species_list, AudioDeterministicESCDataset, \
    ESC50_BACKGROUND_ENVIRONMENTAL_CLASSES_ALL, ESC50_BACKGROUND_ENVIRONMENTAL_CLASSES_AUGMENTS, \
    AudioCappedDeterministicBirdDataset
from torch_siamese import SiameseNetwork

ANALYSIS_SUBDIR = "analysis"


def load_siamese_network(
        model_dir: str,
):
    model_path = os.path.join(model_dir, "siamese_network.pt")

    kwargs, model_state_dict = torch.load(model_path)

    model = SiameseNetwork(**kwargs)

    model.load_state_dict(model_state_dict)
    model = model.to('cuda')

    return model


class TwoInputSequential(torch.nn.Module):
    def __init__(self, layers):
        super(TwoInputSequential, self).__init__()
        self.layers = layers

    def forward(self, input1, input2):
        out = self.layers[0](input1, input2)
        for layer in self.layers[1:]:
            out = layer(out)
        return out


class PairwiseDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, labels):
        super(PairwiseDataset, self).__init__()
        self.inputs = inputs
        self.labels = labels

        assert (len(inputs) == len(labels))

    def __len__(self):
        return len(self.inputs) ** 2

    def __getitem__(self, index):
        index1 = index % len(self.inputs)
        index2 = (index - index1) // len(self.inputs)

        return (index1, self.inputs[index1], self.labels[index1]), (index2, self.inputs[index2], self.labels[index2])


class PairwiseDatasetZip(torch.utils.data.Dataset):
    def __init__(self, inputs1, labels1, inputs2, labels2):
        super(PairwiseDatasetZip, self).__init__()
        self.inputs1 = inputs1
        self.inputs2 = inputs2
        self.labels1 = labels1
        self.labels2 = labels2

        assert (len(inputs1) == len(labels1) and (len(inputs2) == len(labels2)))

    def __len__(self):
        return len(self.inputs1) * len(self.inputs2)

    def __getitem__(self, index):
        index2 = index % len(self.inputs2)
        index1 = (index - index2) // len(self.inputs2)

        return (index1, self.inputs1[index1], self.labels1[index1]), \
            (index2, self.inputs2[index2], self.labels2[index2])


def analyze_test_set_with_train_mixin(
        model_dir: str,
        data_dir: str,
        split: str,
        species_list: List[str],
        classes_out_fn: str,
        min_sns: float = 50,
        mixin_data_dir: str = None,
        mixin_data_split: str = None,
        mixin_data_num_per_class: int = None,
        mixin_species_list: List[str] = None,
        out_fn: str = None
):
    if out_fn is None:
        out_fn = f"siamese_outputs_{split}_all.csv"

    out_dir = os.path.join(model_dir, ANALYSIS_SUBDIR)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_path = os.path.join(out_dir, out_fn)
    out_classes_path = os.path.join(out_dir, classes_out_fn)

    if os.path.isfile(out_path):
        print("All-Pairs-Matrix already exists.. Skipping..")
        return

    siamese_network = load_siamese_network(model_dir)

    siamese_network.eval()

    dataset = AudioDeterministicBirdDataset(
        data_dir,
        split=split,
        min_sns=min_sns,
        species_list=species_list,
        target_encoding=TargetEncoding.CATEGORICAL,
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

    if mixin_data_dir is not None:
        mixin_dataset = AudioCappedDeterministicBirdDataset(
            mixin_data_dir,
            mixin_data_split,
            max_num_per_class=mixin_data_num_per_class,
            min_sns=min_sns,
            dst_sr=44100,
            species_list=mixin_species_list,
            target_encoding=TargetEncoding.CATEGORICAL,
            mel_transforms=torch.nn.Sequential(
                torchvision.transforms.Resize(
                    (128, 384),
                    interpolation=torchvision.transforms.InterpolationMode.BILINEAR
                ),
                torchvision.transforms.Normalize(-30.0118, 12.8596)
            )
        )

        print(f"Mixin in {len(mixin_dataset)} bird samples with {mixin_data_num_per_class} samples per bird class...")

        mixin_dataloader = DataLoader(mixin_dataset, batch_size=1024, num_workers=os.cpu_count())
    else:
        mixin_dataset = None
        mixin_dataloader = None

    dataloader = DataLoader(dataset, batch_size=1024, num_workers=os.cpu_count())

    with torch.no_grad():

        embedding_network = list(siamese_network.children())[0]
        classifier_head = TwoInputSequential(list(siamese_network.children())[1:])

        embeddings = torch.empty(size=(0,)).to('cuda')
        embedding_labels = []

        for batch_idx, (images, labels) in enumerate(tqdm(dataloader)):
            images = images.to('cuda')
            embeddings = torch.cat([embeddings, embedding_network(images)], dim=0)
            embedding_labels.extend(labels)

        if mixin_dataloader is not None:
            for batch_idx, (images, labels) in enumerate(tqdm(mixin_dataloader)):
                images = images.to('cuda')
                embeddings = torch.cat([embeddings, embedding_network(images)], dim=0)
                embedding_labels.extend(labels)

        with open(out_classes_path, "w") as f:
            csv_writer = csv.DictWriter(f, fieldnames=["index", "label", "subset"])
            csv_writer.writeheader()
            rows = [
                {
                    "index": index,
                    "label": embedding_labels[index],
                    "subset": "TEST" if index < len(dataset) else "TRAIN"
                }
                for index in range(len(dataset) + len(mixin_dataset))
            ]
            csv_writer.writerows(rows)

        pair_ds = PairwiseDataset(embeddings.cpu(), embedding_labels)
        pair_ds_loader = DataLoader(pair_ds, batch_size=2 ** 18, num_workers=os.cpu_count())

        with open(out_path, "w") as f:
            csv_writer = csv.DictWriter(f, fieldnames=["i", "j", "pred", "label"])
            csv_writer.writeheader()

            for batch_idx, ((indices1, images1, labels1), (indices2, images2, labels2)) in enumerate(
                    tqdm(pair_ds_loader)):
                images1, images2 = images1.to('cuda'), images2.to('cuda')
                siamese_output = classifier_head(images1, images2)
                # labels_zipped = list(zip(labels1, labels2))
                # indices_zipped = list(zip(indices1, indices2))

                # no generate a csv which looks like
                # (index1, index2, pred, truth) where pred in [0, 1] and truth either 0 or 1

                rows = []

                for num in range(len(siamese_output)):
                    rows.append({
                        "i": int(indices1[num]),
                        "j": int(indices2[num]),
                        "pred": "%.4f" % float(siamese_output[num]),
                        "label": int(labels1[num] == labels2[num])
                    })

                csv_writer.writerows(rows)


def analyze_unseen_birds_and_test_set(
        model_dir: str,
        unseen_birds_data_dir: str,
        classes_out_fn: str,
        test_set_data_dir: str,
        split: str,
        species_list: List[str],
        min_sns: float = 50,
        out_fn: str = None
):
    if out_fn is None:
        out_fn = f"siamese_outputs_{split}_all.csv"

    out_dir = os.path.join(model_dir, ANALYSIS_SUBDIR)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_path = os.path.join(out_dir, out_fn)
    out_classes_path = os.path.join(out_dir, classes_out_fn)

    if os.path.isfile(out_path):
        print("All-Pairs-Matrix already exists.. Skipping..")
        return

    siamese_network = load_siamese_network(model_dir)

    siamese_network.eval()

    dataset = AudioDeterministicBirdDataset(
        unseen_birds_data_dir,
        split="test",
        min_sns=min_sns,
        species_list=None,
        target_encoding=TargetEncoding.CATEGORICAL,
        mel_transforms=torch.nn.Sequential(
            torchvision.transforms.Resize(
                (128, 384),
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR
            ),
            torchvision.transforms.Normalize(-30.0118, 12.8596)
        )
    )

    print("Total Samples in Unseen Birds Test Set: ", len(dataset))
    if len(dataset) == 0:
        return

    test_data_set = AudioDeterministicBirdDataset(
        test_set_data_dir,
        split,
        min_sns=min_sns,
        dst_sr=44100,
        species_list=species_list,
        target_encoding=TargetEncoding.CATEGORICAL,
        mel_transforms=torch.nn.Sequential(
            torchvision.transforms.Resize(
                (128, 384),
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR
            ),
            torchvision.transforms.Normalize(-30.0118, 12.8596)
        )
    )

    print(f"Test Set Size {len(test_data_set)} bird samples...")

    test_set_dataloader = DataLoader(test_data_set, batch_size=1024, num_workers=os.cpu_count())

    dataloader = DataLoader(dataset, batch_size=1024, num_workers=os.cpu_count())

    with torch.no_grad():

        embedding_network = list(siamese_network.children())[0]
        classifier_head = TwoInputSequential(list(siamese_network.children())[1:])

        embeddings = torch.empty(size=(0,)).to('cuda')
        embedding_labels = []

        for batch_idx, (images, labels) in enumerate(tqdm(dataloader)):
            images = images.to('cuda')
            embeddings = torch.cat([embeddings, embedding_network(images)], dim=0)
            embedding_labels.extend(labels)

        for batch_idx, (images, labels) in enumerate(tqdm(test_set_dataloader)):
            images = images.to('cuda')
            embeddings = torch.cat([embeddings, embedding_network(images)], dim=0)
            embedding_labels.extend(labels)

        with open(out_classes_path, "w") as f:
            csv_writer = csv.DictWriter(f, fieldnames=["index", "label", "subset"])
            csv_writer.writeheader()
            rows = [
                {
                    "index": index,
                    "label": embedding_labels[index],
                    "subset": "UNSEEN" if index < len(dataset) else "TEST"
                }
                for index in range(len(dataset) + len(test_data_set))
            ]
            csv_writer.writerows(rows)

        pair_ds = PairwiseDataset(embeddings.cpu(), embedding_labels)
        pair_ds_loader = DataLoader(pair_ds, batch_size=2 ** 18, num_workers=os.cpu_count())

        with open(out_path, "w") as f:
            csv_writer = csv.DictWriter(f, fieldnames=["i", "j", "pred", "label"])
            csv_writer.writeheader()

            for batch_idx, ((indices1, images1, labels1), (indices2, images2, labels2)) in enumerate(
                    tqdm(pair_ds_loader)):
                images1, images2 = images1.to('cuda'), images2.to('cuda')
                siamese_output = classifier_head(images1, images2)

                # no generate a csv which looks like
                # (index1, index2, pred, truth) where pred in [0, 1] and truth either 0 or 1

                rows = []

                for num in range(len(siamese_output)):
                    rows.append({
                        "i": int(indices1[num]),
                        "j": int(indices2[num]),
                        "pred": "%.4f" % float(siamese_output[num]),
                        "label": int(labels1[num] == labels2[num])
                    })

                csv_writer.writerows(rows)


def analyse_esc50_test_set_and_birds_test_set(
        model_dir: str,
        esc50_data_dir: str,
        classes_out_fn: str,
        test_set_data_dir: str,
        split: str,
        species_list: List[str],
        min_sns: float = 50,
        out_fn: str = None
):
    if out_fn is None:
        out_fn = f"siamese_outputs_{split}_all.csv"

    out_dir = os.path.join(model_dir, ANALYSIS_SUBDIR)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_path = os.path.join(out_dir, out_fn)
    out_classes_path = os.path.join(out_dir, classes_out_fn)

    if os.path.isfile(out_path):
        print("All-Pairs-Matrix already exists.. Skipping..")
        return

    siamese_network = load_siamese_network(model_dir)

    siamese_network.eval()

    dataset = AudioDeterministicESCDataset(
        directory=esc50_data_dir,
        dst_sr=44100,
        classes=sorted(list(set(ESC50_BACKGROUND_ENVIRONMENTAL_CLASSES_ALL)
                            .difference(ESC50_BACKGROUND_ENVIRONMENTAL_CLASSES_AUGMENTS))),
        target_encoding=TargetEncoding.CATEGORICAL,
        mel_transforms=torch.nn.Sequential(
            torchvision.transforms.Resize(
                (128, 384),
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR
            ),
            torchvision.transforms.Normalize(-30.0118, 12.8596)
        )
    )

    print("Total Samples in ESC50 Test Set: ", len(dataset))
    if len(dataset) == 0:
        return

    test_data_set = AudioDeterministicBirdDataset(
        test_set_data_dir,
        split,
        min_sns=min_sns,
        dst_sr=44100,
        species_list=species_list,
        target_encoding=TargetEncoding.CATEGORICAL,
        mel_transforms=torch.nn.Sequential(
            torchvision.transforms.Resize(
                (128, 384),
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR
            ),
            torchvision.transforms.Normalize(-30.0118, 12.8596)
        )
    )

    print(f"Test Set Size {len(test_data_set)} bird samples...")

    test_set_dataloader = DataLoader(test_data_set, batch_size=1024, num_workers=os.cpu_count())

    dataloader = DataLoader(dataset, batch_size=1024, num_workers=os.cpu_count())

    with torch.no_grad():

        embedding_network = list(siamese_network.children())[0]
        classifier_head = TwoInputSequential(list(siamese_network.children())[1:])

        embeddings = torch.empty(size=(0,)).to('cuda')
        embedding_labels = []

        for batch_idx, (images, labels) in enumerate(tqdm(dataloader)):
            images = images.to('cuda')
            embeddings = torch.cat([embeddings, embedding_network(images)], dim=0)
            embedding_labels.extend(labels)

        for batch_idx, (images, labels) in enumerate(tqdm(test_set_dataloader)):
            images = images.to('cuda')
            embeddings = torch.cat([embeddings, embedding_network(images)], dim=0)
            embedding_labels.extend(labels)

        with open(out_classes_path, "w") as f:
            csv_writer = csv.DictWriter(f, fieldnames=["index", "label", "subset"])
            csv_writer.writeheader()
            rows = [
                {
                    "index": index,
                    "label": embedding_labels[index],
                    "subset": "ESC50" if index < len(dataset) else "TEST"
                }
                for index in range(len(dataset) + len(test_data_set))
            ]
            csv_writer.writerows(rows)

        pair_ds = PairwiseDataset(embeddings.cpu(), embedding_labels)
        pair_ds_loader = DataLoader(pair_ds, batch_size=2 ** 18, num_workers=os.cpu_count())

        with open(out_path, "w") as f:
            csv_writer = csv.DictWriter(f, fieldnames=["i", "j", "pred", "label"])
            csv_writer.writeheader()

            for batch_idx, ((indices1, images1, labels1), (indices2, images2, labels2)) in enumerate(
                    tqdm(pair_ds_loader)):
                images1, images2 = images1.to('cuda'), images2.to('cuda')
                siamese_output = classifier_head(images1, images2)
                # labels_zipped = list(zip(labels1, labels2))
                # indices_zipped = list(zip(indices1, indices2))

                # no generate a csv which looks like
                # (index1, index2, pred, truth) where pred in [0, 1] and truth either 0 or 1

                rows = []

                for num in range(len(siamese_output)):
                    rows.append({
                        "i": int(indices1[num]),
                        "j": int(indices2[num]),
                        "pred": "%.4f" % float(siamese_output[num]),
                        "label": int(labels1[num] == labels2[num])
                    })

                csv_writer.writerows(rows)


def analyse_all_pairs_esc50_noise(
        model_dir: str,
        esc_data_dir: str,
        mixin_data_dir: str = None,
        mixin_data_split: str = None,
        mixin_data_num_per_class: int = None,
        mixin_species_list: List[str] = None,
        esc_classes: List[str] = None,
        out_fn: str = "test.csv"
):
    out_dir = os.path.join(model_dir, ANALYSIS_SUBDIR)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_path = os.path.join(out_dir, out_fn)

    if os.path.isfile(out_path):
        print("All-Pairs-Matrix already exists.. Skipping..")
        return

    siamese_network = load_siamese_network(model_dir)

    siamese_network.eval()

    esc_dataset = AudioDeterministicESCDataset(
        esc_data_dir,
        dst_sr=44100,
        classes=esc_classes,
        target_encoding=TargetEncoding.CATEGORICAL,
        mel_transforms=torch.nn.Sequential(
            torchvision.transforms.Resize(
                (128, 384),
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR
            ),
            torchvision.transforms.Normalize(-30.0118, 12.8596)
        )
    )

    print("Total Samples in ESC Set: ", len(esc_dataset))
    if len(esc_dataset) == 0:
        return
    esc_dataloader = DataLoader(esc_dataset, batch_size=1024, num_workers=os.cpu_count())

    if mixin_data_dir is not None:
        mixin_dataset = AudioCappedDeterministicBirdDataset(
            mixin_data_dir,
            mixin_data_split,
            max_num_per_class=mixin_data_num_per_class,
            min_sns=50,
            dst_sr=44100,
            species_list=mixin_species_list,
            target_encoding=TargetEncoding.CATEGORICAL,
            mel_transforms=torch.nn.Sequential(
                torchvision.transforms.Resize(
                    (128, 384),
                    interpolation=torchvision.transforms.InterpolationMode.BILINEAR
                ),
                torchvision.transforms.Normalize(-30.0118, 12.8596)
            )
        )

        print(f"Mixin in {len(mixin_dataset)} bird samples with {mixin_data_num_per_class} samples per bird class...")

        mixin_dataloader = DataLoader(mixin_dataset, batch_size=1024, num_workers=os.cpu_count())
    else:
        mixin_dataloader = None

    with torch.no_grad():

        embedding_network = list(siamese_network.children())[0]
        classifier_head = TwoInputSequential(list(siamese_network.children())[1:])

        embeddings = torch.empty(size=(0,)).to('cuda')
        embedding_labels = []

        for batch_idx, (images, labels) in enumerate(tqdm(esc_dataloader)):
            images = images.to('cuda')
            embeddings = torch.cat([embeddings, embedding_network(images)], dim=0)
            embedding_labels.extend(labels)

        if mixin_dataloader is not None:
            for batch_idx, (images, labels) in enumerate(tqdm(mixin_dataloader)):
                images = images.to('cuda')
                embeddings = torch.cat([embeddings, embedding_network(images)], dim=0)
                embedding_labels.extend(labels)

        pair_ds = PairwiseDataset(embeddings.cpu(), embedding_labels)
        pair_ds_loader = DataLoader(pair_ds, batch_size=2 ** 18, num_workers=os.cpu_count())

        with open(out_path, "w") as f:
            csv_writer = csv.DictWriter(f, fieldnames=["i", "j", "pred", "label", "isubset", "jsubset"])
            csv_writer.writeheader()

            for batch_idx, ((indices1, images1, labels1), (indices2, images2, labels2)) in enumerate(
                    tqdm(pair_ds_loader)):
                images1, images2 = images1.to('cuda'), images2.to('cuda')
                siamese_output = classifier_head(images1, images2)
                # labels_zipped = list(zip(labels1, labels2))
                # indices_zipped = list(zip(indices1, indices2))

                # no generate a csv which looks like
                # (index1, index2, pred, truth) where pred in [0, 1] and truth either 0 or 1

                rows = []

                for num in range(len(siamese_output)):
                    rows.append({
                        "i": int(indices1[num]),
                        "j": int(indices2[num]),
                        "pred": "%.4f" % float(siamese_output[num]),
                        "label": int(labels1[num] == labels2[num]),
                        "isubset": "ESC50" if indices1[num] < len(esc_dataset) else "Mixin",
                        "jsubset": "ESC50" if indices2[num] < len(esc_dataset) else "Mixin"
                    })

                csv_writer.writerows(rows)


def analyze_test_set_only(
        model_dir: str,
        data_dir: str,
        classes_out_fn: str,
        split: str,
        species_list: Union[List[str], None],
        min_sns: float = 50,
        out_fn: str = None
):
    if out_fn is None:
        out_fn = f"siamese_outputs_{split}_all.csv"

    out_dir = os.path.join(model_dir, ANALYSIS_SUBDIR)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_path = os.path.join(out_dir, out_fn)
    out_classes_path = os.path.join(out_dir, classes_out_fn)

    if os.path.isfile(out_path):
        print("All-Pairs-Matrix already exists.. Skipping..")
        return

    siamese_network = load_siamese_network(model_dir)

    siamese_network.eval()

    dataset = AudioDeterministicBirdDataset(
        data_dir,
        split,
        min_sns=min_sns,
        dst_sr=44100,
        species_list=species_list,
        target_encoding=TargetEncoding.CATEGORICAL,
        mel_transforms=torch.nn.Sequential(
            torchvision.transforms.Resize(
                (128, 384),
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR
            ),
            torchvision.transforms.Normalize(-30.0118, 12.8596)
        )
    )

    print(f"Test Set Size {len(dataset)} bird samples...")
    dataloader = DataLoader(dataset, batch_size=1024, num_workers=os.cpu_count())

    with torch.no_grad():
        embedding_network = list(siamese_network.children())[0]
        classifier_head = TwoInputSequential(list(siamese_network.children())[1:])

        embeddings = torch.empty(size=(0,)).to('cuda')
        embedding_labels = []

        for batch_idx, (images, labels) in enumerate(tqdm(dataloader)):
            images = images.to('cuda')
            embeddings = torch.cat([embeddings, embedding_network(images)], dim=0)
            embedding_labels.extend(labels)

        with open(out_classes_path, "w") as f:
            csv_writer = csv.DictWriter(f, fieldnames=["index", "label"])
            csv_writer.writeheader()
            rows = [
                {
                    "index": index,
                    "label": embedding_labels[index]
                }
                for index in range(len(dataset))
            ]
            csv_writer.writerows(rows)

        pair_ds = PairwiseDataset(embeddings.cpu(), embedding_labels)
        pair_ds_loader = DataLoader(pair_ds, batch_size=2 ** 18, num_workers=os.cpu_count())

        with open(out_path, "w") as f:
            csv_writer = csv.DictWriter(f, fieldnames=["i", "j", "pred", "label"])
            csv_writer.writeheader()

            for batch_idx, ((indices1, images1, labels1), (indices2, images2, labels2)) in enumerate(
                    tqdm(pair_ds_loader)):
                images1, images2 = images1.to('cuda'), images2.to('cuda')
                siamese_output = classifier_head(images1, images2)

                rows = []

                for num in range(len(siamese_output)):
                    rows.append({
                        "i": int(indices1[num]),
                        "j": int(indices2[num]),
                        "pred": "%.4f" % float(siamese_output[num]),
                        "label": int(labels1[num] == labels2[num])
                    })

                csv_writer.writerows(rows)


def analyze_classifier_protocol_nearest_train_cluster(
        model_dir: str,
        data_dir: str,
        species_list: List[str],
        min_sns: float = 50,
        out_fn: str = None
):
    out_dir = os.path.join(model_dir, ANALYSIS_SUBDIR)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_path = os.path.join(out_dir, out_fn)

    if os.path.isfile(out_path):
        print("All-Pairs-Matrix already exists.. Skipping..")
        return

    siamese_network = load_siamese_network(model_dir)

    siamese_network.eval()

    test_dataset = AudioDeterministicBirdDataset(
        data_dir,
        split="test",
        min_sns=min_sns,
        species_list=species_list,
        target_encoding=TargetEncoding.CATEGORICAL,
        mel_transforms=torch.nn.Sequential(
            torchvision.transforms.Resize(
                (128, 384),
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR
            ),
            torchvision.transforms.Normalize(-30.0118, 12.8596)
        )
    )

    print("Total Samples in Test Set: ", len(test_dataset))
    if len(test_dataset) == 0:
        return

    train_dataset = AudioDeterministicBirdDataset(
        data_dirs=[data_dir],
        split="train",
        min_sns=min_sns,
        dst_sr=44100,
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

    train_dataloader = DataLoader(train_dataset, batch_size=1024, num_workers=os.cpu_count())

    test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=os.cpu_count())

    with torch.no_grad():

        embedding_network = list(siamese_network.children())[0]
        classifier_head = TwoInputSequential(list(siamese_network.children())[1:])

        embeddings = torch.empty(size=(0,)).to('cuda')
        embedding_labels = torch.empty(size=(0,)).to('cuda')

        print("Calculate train embedding vectors...")
        for batch_idx, (images, labels) in enumerate(tqdm(train_dataloader)):
            images = images.to('cuda')
            embeddings = torch.cat([embeddings, embedding_network(images)], dim=0)
            embedding_labels = torch.cat([embedding_labels, labels.cuda()], dim=0)

        test_embedding_true_labels = []
        test_embedding_pred_labels = []
        for batch_idx, (images, labels) in enumerate(tqdm(test_dataloader)):
            images = images.to('cuda')
            test_embedding = embedding_network(images)
            test_embedding = torch.repeat_interleave(test_embedding, embeddings.shape[0], 0)
            test_embedding_true_labels.extend(labels)

            predictions = (classifier_head(embeddings, test_embedding) + classifier_head(test_embedding,
                                                                                         embeddings)).squeeze() / 2

            best_species = "background"
            best_cost_assignment = 0

            # collect all predictions by where the last dimension of embedding_labels is 1
            for species_index in range(len(train_dataset.species)):
                projected_tensor = embedding_labels[:, species_index]
                projected_indices = projected_tensor.nonzero().squeeze()

                predictions_projected = predictions[projected_indices]

                # clip
                predictions_projected = torch.clamp(predictions_projected, min=1e-6, max=1 - 1e-6)

                # p -> 0 => cost > 0
                # p -> 1 => cost < 0
                cost = torch.log((1 - predictions_projected) / predictions_projected).sum().item()

                if cost > 0:
                    # do nothing and best species index is still 0 (background)
                    pass
                if cost < best_cost_assignment:
                    best_cost_assignment = cost
                    best_species = train_dataset.species[species_index]

            test_embedding_pred_labels.append(best_species)

        with open(out_path, "w") as f:
            csv_writer = csv.DictWriter(f, fieldnames=["index", "predIdx", "trueIdx", "predLabel", "trueLabel"])
            csv_writer.writeheader()
            rows = []
            for index in range(len(test_dataset)):
                true_label = test_embedding_true_labels[index]
                pred_label = test_embedding_pred_labels[index]

                true_label_idx = 0 if true_label not in train_dataset.species else (
                            train_dataset.species.index(true_label) + 1)
                pred_label_idx = 0 if pred_label not in train_dataset.species else (
                            train_dataset.species.index(pred_label) + 1)

                rows.append({
                    "index": index,
                    "predIdx": pred_label_idx,
                    "trueIdx": true_label_idx,
                    "predLabel": pred_label,
                    "trueLabel": true_label
                })

            csv_writer.writerows(rows)


def main():
    models_dir = os.path.join(config.MODELS_DIR, "similarity", "model-3600")

    data_dir = "./data/split-3600"

    unseen_data_dir = "./data/unseen"

    esc_dir = config.ESC_DIR

    species_list = read_species_list("./species-3600.txt")

    analyze_classifier_protocol_nearest_train_cluster(
        model_dir=models_dir,
        min_sns=config.MIN_SNS,
        out_fn="classifier_protocol.csv",
        species_list=species_list,
        data_dir=data_dir
    )

    analyze_test_set_only(
        model_dir=models_dir,
        split="test",
        out_fn="unseen_only.csv",
        classes_out_fn="unseen_only_classes.csv",
        data_dir=unseen_data_dir,
        species_list=None,
        min_sns=config.MIN_SNS,
    )
    analyze_test_set_only(
        model_dir=models_dir,
        split="test",
        out_fn="test_set_only.csv",
        classes_out_fn="test_set_only_classes.csv",
        data_dir=data_dir,
        species_list=species_list,
        min_sns=config.MIN_SNS,
    )

    analyze_unseen_birds_and_test_set(
        model_dir=models_dir,
        test_set_data_dir=data_dir,
        species_list=species_list,
        classes_out_fn="unseen_and_test_classes.csv",
        unseen_birds_data_dir="./data/unseen",
        out_fn=f"unseen_and_test_separated_classes.csv",
        split="test",
        min_sns=config.MIN_SNS
    )

    analyse_esc50_test_set_and_birds_test_set(
        model_dir=models_dir,
        species_list=species_list,
        classes_out_fn="esc50_and_test_classes.csv",
        esc50_data_dir=esc_dir,
        min_sns=config.MIN_SNS,
        split="test",
        out_fn=f"esc50_and_test.csv",
        test_set_data_dir=data_dir
    )

    analyze_test_set_with_train_mixin(
        model_dir=models_dir,
        min_sns=config.MIN_SNS,
        classes_out_fn="test_with_train128_classes.csv",
        out_fn="test_with_train128_separated_classes.csv",
        mixin_data_dir=data_dir,
        mixin_data_split="train",
        species_list=species_list,
        mixin_data_num_per_class=128,
        data_dir=data_dir,
        mixin_species_list=species_list,
        split="test"
    )


if __name__ == "__main__":
    main()
