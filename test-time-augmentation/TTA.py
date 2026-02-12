
from pathlib import Path
from datetime import datetime
import sys
import json 
from tqdm import tqdm

import torchio as tio
import torch
import numpy as np
import monai 
import nibabel as nib

from torch.utils.data import DataLoader

from utils.load_model import load
from utils.transforms import tta 
# from utils.get_dice import get_dice_score
from utils.get_subjects import get_subjects
from utils.transforms import preprocess

import os

weights_id = str(sys.argv[1])

weigts_dir = Path("./weights")
data_infos_path = str(sorted(weigts_dir.glob('**/*'+weights_id+'/dataset*.json'))[0])
config_file_path = str(sorted(weigts_dir.glob('**/*'+weights_id+'/config*.json'))[0])
config_file = config_file_path.split("/")[-1].split(".json")[0]
data_info_file = data_infos_path.split("/")[-1].split(".json")[0]
print(f"######## Inference with config {config_file_path} and data {data_infos_path} for weights {weights_id}")
with open(config_file_path) as f:
        ctx = json.load(f)
        num_workers = ctx["num_workers"]
        num_epochs = ctx["num_epochs"]
        task = ctx["experiment_name"]
        lr = ctx["initial_lr"]
        seed = ctx["seed"]
        net_model = ctx["net_model"]
        batch_size = ctx["batch_size"]
        dropout = ctx["dropout"]
        loss_type = ctx['loss_type']
        channels = ctx["channels"]
        n_layers = len(channels)
        train_val_ratio = ctx["train_val_ratio"]
        if ctx["patch"]:
            patch_size = ctx["patch_size"]
            queue_length = ctx["queue_length"]
            samples_per_volume = ctx["samples_per_volume"] 

with open(data_infos_path) as f:
    data_info = json.load(f)
    channel = data_info["channel_names"]["0"]
    rootdir_train_img = data_info["rootdir_train_img"]
    rootdir_train_labels = data_info["rootdir_train_labels"]
    rootdir_test_img = data_info["rootdir_test-cap"]
    rootdir_test_labels = data_info["rootdir_test_labels-cap"]
    rootdir_test_brain_mask = data_info["rootdir_test_brain_mask"]
    rootdir_train_brain_mask = data_info["rootdir_train_brain_mask"]
    suffixe_img_train = data_info["suffixe_img-train"]
    suffixe_labels_train = data_info["suffixe_labels-train"]
    suffixe_img_test = data_info["suffixe_img-test"]
    suffixe_labels_test = data_info["suffixe_labels-test"]
    suffixe_brain_mask_train = data_info["suffixe_brain_mask-train"]
    suffixe_brain_mask_test = data_info["suffixe_brain_mask-test"]
    num_classes = len(data_info["labels"])
    file_ending = data_info["file_ending"]
    subsample = data_info["subset"]
    labels_names = list(data_info["labels"].keys())
print(f"{num_classes} classes : {labels_names}")

sample = ""
if subsample:
    sample = "_subset"

n_segmentations = 10

current_dateTime = datetime.now()
id_run = config_file + sample + "_weights-" + weights_id + "_" + str(current_dateTime.day) + "-" + str(current_dateTime.month) + "-" + str(current_dateTime.year) + "-" + str(current_dateTime.hour) + str(current_dateTime.minute) + str(current_dateTime.second) 
model_weights_path = str(sorted(weigts_dir.glob('**/*'+weights_id+'.pth'))[0])
out_path=f"/home/emma/Projets/stroke_lesion_segmentation_v2/out-predictions/TTA/{id_run}/"
print(f"Inference with weights : {model_weights_path}")

##############
#   Devices  #
##############

if torch.cuda.is_available():
    [print() for i in range(torch.cuda.device_count())]
    [print(f"Available GPUs : \n{i} : {torch.cuda.get_device_name(i)}") for i in range(torch.cuda.device_count())]
    device = "cuda" 
else:
    device = "cpu"
print(f"device used : {device}")


#################
#   LOAD MODEL  #
#################

model = load(model_weights_path,net_model, lr, dropout, loss_type, num_classes, channels, num_epochs)
model.eval()   

##############
#   IMAGES    #
##############
print("\n# DATA PATH : \n")

img_dir=Path(rootdir_test_img)
brain_masks_dir=Path(rootdir_test_brain_mask)
labels_dir=Path(rootdir_test_labels)

print(f"Test images in : {img_dir}")
print(f"Test labels in : {labels_dir}")
print(f"Test brain masks in : {brain_masks_dir}")

img_paths = sorted(img_dir.glob('**/*'+suffixe_img_test+file_ending))
test_label_paths = sorted(labels_dir.glob('**/*'+suffixe_labels_test+file_ending))
brain_mask_paths = sorted(brain_masks_dir.glob('**/*'+suffixe_brain_mask_test+file_ending))

assert len(img_paths) == len(test_label_paths)

subjects = get_subjects(img_paths, label_paths=test_label_paths, subsample=False, brain_mask_paths=brain_mask_paths)[1:]

subjects_dataset = tio.SubjectsDataset(subjects)

def get_variance(Y):
    return np.std(Y, axis=0)

#################
#   INFERENCE   #
#################
      
def inference(subject, metric=False, patch=False):
    get_dice = monai.metrics.DiceMetric(include_background=False, reduction="none")
    get_hd = monai.metrics.HausdorffDistanceMetric(include_background=False, reduction="none", percentile=95)

    if patch:
        grid_sampler = tio.inference.GridSampler(subject, patch_size, patch_overlap=32)
        aggregator = tio.inference.GridAggregator(grid_sampler)
        loader = DataLoader(grid_sampler)
        # subject.clear_history()
    else:
        loader = subject

    with torch.no_grad():
        if patch:
            for batch in tqdm(loader, unit='batch'):
                input = batch['t1'][tio.DATA]
                logits = model.net(input)
                labels = logits.argmax(dim=tio.CHANNELS_DIMENSION, keepdim=True)
                locations = batch[tio.LOCATION]
                aggregator.add_batch(labels, locations)
            output_tensor = aggregator.get_output_tensor().to(torch.int64).unsqueeze(axis=0)
        else:
            input = subject['t1'][tio.DATA].unsqueeze(axis=0)
            logits = model.net(input)
            output_tensor = logits.argmax(dim=tio.CHANNELS_DIMENSION, keepdim=True)

    if metric:
        gt_tensor = subject['seg'][tio.DATA].unsqueeze(axis=0)
        outputs_one_hot = torch.nn.functional.one_hot(output_tensor, num_classes=num_classes).squeeze(axis=1)
        outputs_one_hot = outputs_one_hot.permute(0, 4, 1, 2, 3)

        get_dice(outputs_one_hot.to(model.device), gt_tensor.to(model.device))
        get_hd(outputs_one_hot.to(model.device), gt_tensor.to(model.device))

        dice = get_dice.aggregate().cpu().numpy()[0]
        hd = get_hd.aggregate().cpu().numpy()[0]
        surface_dice = monai.metrics.compute_surface_dice(outputs_one_hot.to(model.device), gt_tensor.to(model.device), class_thresholds=[1]).cpu().numpy()[0]

        print(f"DICE: {dice[0]}, HD: {hd[0]}, Surface DICE: {surface_dice[0]}")
        get_dice.reset()
        get_hd.reset()

    # Extract lesion class and create binary mask
    outputs_one_hot = torch.nn.functional.one_hot(output_tensor, num_classes=num_classes).squeeze(axis=1)
    outputs_one_hot = outputs_one_hot.permute(0, 4, 1, 2, 3)
    lesion_mask = outputs_one_hot[0, 1, ...].unsqueeze(0)  # Shape: (1, H, W, D)
    binary_mask = (lesion_mask > 0.5).to(torch.int64)

    # Create a tio.LabelMap for the binary mask
    pred = tio.LabelMap(tensor=binary_mask, affine=subject.t1.affine, type=tio.LABEL)

    subject.add_image(pred, "prediction")
    return subject



###########################
#   DATA TRANFORMATIONS   #
###########################

def get_tta(segmentations, subject, i, out_file):
    preds = np.stack(segmentations, axis=0).mean(axis=0)
    threshold = 0.5
    tta_prediction = (preds > threshold).astype(np.uint8)
    variance_uncertainty_estimation_image = get_variance(np.stack(segmentations, axis=0))

    # Create tio.LabelMap for prediction and variance
    tta_pred = tio.LabelMap(tensor=tta_prediction, affine=subject.t1.affine)
    var_uncertainty = tio.ScalarImage(tensor=variance_uncertainty_estimation_image, affine=subject.t1.affine)

    # Save using tio's save method

    tta_pred.save(os.path.join(out_file, f"{subject.subject}_tta_pred_n{i+1}.nii.gz"))
    var_uncertainty.save(os.path.join(out_file, f"{subject.subject}_vue_n{i+1}.nii.gz"))


native_space_subjects = []
native_space_subjects_segmentations = []

variance_uncertainty_estimation_images = []
for subject in subjects_dataset.dry_iter():
    segmentations = []
    print(f"Subject : {subject.subject}")
    for i in tqdm(range(n_segmentations)):
        transforms = tio.Compose([preprocess(brain_mask='brain_mask'), tta()]) # /!\ Resample is not invertible, cannot be used with tta. 
        transformed_subject = transforms(subject)

        # Run inference on the transformed subject
        subject_with_pred = inference(transformed_subject, metric=True, patch=ctx["patch"])

        # Apply inverse transforms to get the prediction in native space
        native_space_subject = subject_with_pred.apply_inverse_transform(image_interpolation='linear')

        # Append the prediction data for TTA
        segmentations.append(native_space_subject.prediction.data.numpy())

        # Save the images in native space
        out_file = f"./out-predictions/TTA/{id_run}/{subject.subject}"
        os.makedirs(out_file, exist_ok=True)

        # Save using tio's save method to preserve affine and metadata
        native_space_subject.prediction.save(f"{out_file}/{subject.subject}_pred.nii.gz")
        native_space_subject.seg.save(f"{out_file}/{subject.subject}_seg.nii.gz")
        native_space_subject.t1.save(f"{out_file}/{subject.subject}_t1w.nii.gz")

        if i == 4 or i == 9:
            get_tta(segmentations, native_space_subject, i, out_file)


