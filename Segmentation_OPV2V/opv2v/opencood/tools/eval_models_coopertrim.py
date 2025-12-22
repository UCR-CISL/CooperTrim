import argparse
import statistics
import time
import os
from glob import glob

import torch
from torch.utils.data import DataLoader

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, infrence_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils.seg_utils import cal_iou_training


def test_parser():
    parser = argparse.ArgumentParser(description="Evaluate models in subdirectories")
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Parent directory containing subdirectories with models and config files')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save evaluation results')
    parser.add_argument('--model_type', type=str, default='dynamic',
                        help='dynamic or static prediction')
    opt = parser.parse_args()
    return opt


def evaluate_model(model_path, hypes, opt, device):
    print(f"Evaluating model: {model_path}")
    opencood_dataset = build_dataset(hypes, visualize=True, train=False)
    data_loader = DataLoader(opencood_dataset,
                             batch_size=1,
                             num_workers=10,
                             collate_fn=opencood_dataset.collate_batch,
                             shuffle=False,
                             pin_memory=False,
                             drop_last=False)

    model = train_utils.create_model(hypes)
    model.to(device)

    print('Loading Model from checkpoint')
    _, model = train_utils.load_saved_model(model_path, model)
    model.eval()

    dynamic_ave_iou = []
    static_ave_iou = []
    lane_ave_iou = []

    for i, batch_data in enumerate(data_loader):
        print(f"Processing batch {i}")
        with torch.no_grad():
            torch.cuda.synchronize()
            batch_data = train_utils.to_device(batch_data, device)
            output_dict = model(batch_data['ego'])

            # Post-process and visualize
            output_dict = opencood_dataset.post_process(batch_data['ego'], output_dict)
            infrence_utils.camera_inference_visualization(output_dict, batch_data, model_path, i, opt.model_type)

            # Calculate IoU
            iou_dynamic, iou_static = cal_iou_training(batch_data, output_dict)
            static_ave_iou.append(iou_static[1])
            dynamic_ave_iou.append(iou_dynamic[1])
            lane_ave_iou.append(iou_static[2])

    # Compute mean IoU values
    static_ave_iou = statistics.mean(static_ave_iou)
    dynamic_ave_iou = statistics.mean(dynamic_ave_iou)
    lane_ave_iou = statistics.mean(lane_ave_iou)

    print(f"Model: {model_path}")
    print('Road IoU: %f' % static_ave_iou)
    print('Lane IoU: %f' % lane_ave_iou)
    print('Dynamic IoU: %f' % dynamic_ave_iou)

    return static_ave_iou, dynamic_ave_iou, lane_ave_iou


def main():
    opt = test_parser()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Ensure output directory exists
    os.makedirs(opt.output_dir, exist_ok=True)

    # Find all subdirectories in the eval_scripts directory
    subdirectories = [os.path.join(opt.model_dir, d) for d in os.listdir(opt.model_dir)
                      if os.path.isdir(os.path.join(opt.model_dir, d))]

    results = []
    print(f"Found {len(subdirectories)} subdirectories in {opt.model_dir}")
    for subdirectory in subdirectories:
        # Locate the hypes.yaml and .pth file in the subdirectory
        print(f"Processing model for: {subdirectory}")
        hypes_path = os.path.join(subdirectory, "config.yaml")
        model_path = subdirectory

        if not os.path.exists(hypes_path):
            print(f"Skipping {subdirectory}: config.yaml not found")
            continue
        # if len(model_files) == 0:
        #     print(f"Skipping {subdirectory}: No .pth model file found")
        #     continue


        # Load the configuration file
        print(f"Loading configuration from {hypes_path}")
        hypes = yaml_utils.load_yaml(hypes_path, opt)

        # Evaluate the model
        static_iou, dynamic_iou, lane_iou = evaluate_model(model_path, hypes, opt, device)

        # Save results for the current model
        results.append({
            'model_dir': subdirectory,
            'static_iou': static_iou,
            'dynamic_iou': dynamic_iou,
            'lane_iou': lane_iou
        })

    # Save all results to a file
    results_file = os.path.join(opt.output_dir, 'evaluation_results_cobevt_st.txt')
    with open(results_file, 'w') as f:
        for res in results:
            f.write(f"Model: {res['model_dir']}\n")
            f.write(f"Road IoU: {res['static_iou']:.4f}\n")
            f.write(f"Lane IoU: {res['lane_iou']:.4f}\n")
            f.write(f"Dynamic IoU: {res['dynamic_iou']:.4f}\n")
            f.write("\n")

    print(f"Results saved to {results_file}")


if __name__ == '__main__':
    main()
