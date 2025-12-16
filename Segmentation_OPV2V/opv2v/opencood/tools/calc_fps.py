import argparse
import statistics
import time
import torch
from torch.utils.data import DataLoader

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, infrence_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils.seg_utils import cal_iou_training


def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Continued training path')
    parser.add_argument('--model_type', type=str, default='dynamic',
                        help='dynamic or static prediction')
    opt = parser.parse_args()
    return opt


def main():
    opt = test_parser()
    hypes = yaml_utils.load_yaml(None, opt)

    print('Dataset Building')
    opencood_dataset = build_dataset(hypes, visualize=True, train=False)
    data_loader = DataLoader(opencood_dataset,
                             batch_size=1,
                             num_workers=10,
                             collate_fn=opencood_dataset.collate_batch,
                             shuffle=False,
                             pin_memory=False,
                             drop_last=False)

    print('Creating Model')
    model = train_utils.create_model(hypes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # We assume GPU is necessary
    if torch.cuda.is_available():
        model.to(device)

    print('Loading Model from checkpoint')
    saved_path = opt.model_dir
    _, model = train_utils.load_saved_model(saved_path, model)
    model.eval()

    dynamic_ave_iou = []
    static_ave_iou = []
    lane_ave_iou = []
    fps_list = []  # List to store FPS for each batch

    # Open the file for writing dynamic IOU values
    with open("/home/csgrad/smukh039/AutoNetworkingRL/CoBEVT_AutoNet/opv2v/dumps_channel_select/final_cobevt_dyn_channel_select.txt", "w") as f:
        for i, batch_data in enumerate(data_loader):
            print(f"Processing batch {i}")
            with torch.no_grad():
                torch.cuda.synchronize()  # Ensure all GPU operations are complete before starting timing
                
                # Create CUDA events for precise timing
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                
                # Record the start time
                start_event.record()
                
                batch_data = train_utils.to_device(batch_data, device)
                # Shilpa select threshold
                # output_dict, select_threshold, percentage_selected = model(batch_data['ego'], 0)
                output_dict = model(batch_data['ego'])
                
                # Record the end time
                end_event.record()
                
                # Synchronize to ensure timing is accurate
                torch.cuda.synchronize()
                
                # Calculate elapsed time in seconds (CUDA event timing is in milliseconds, so divide by 1000)
                elapsed_time = start_event.elapsed_time(end_event) / 1000.0
                
                # Calculate FPS for this batch (batch_size=1, so FPS = 1 / time)
                fps = 1.0 / elapsed_time if elapsed_time > 0 else 0
                fps_list.append(fps)
                print(f"Batch {i} - Inference Time: {elapsed_time:.4f} seconds, FPS: {fps:.2f}")
                
                # Visualization purpose
                output_dict = opencood_dataset.post_process(batch_data['ego'], output_dict)
                infrence_utils.camera_inference_visualization(output_dict,
                                                              batch_data,
                                                              saved_path,
                                                              i,
                                                              opt.model_type)

                iou_dynamic, iou_static = cal_iou_training(batch_data, output_dict)
                static_ave_iou.append(iou_static[1])
                dynamic_ave_iou.append(iou_dynamic[1])
                lane_ave_iou.append(iou_static[2])

                # Write dynamic IOU to file
                dyniou = (iou_dynamic[1] * 100)
                line_to_write = f"({i},{dyniou:.4f})\n"
                f.write(line_to_write)

    # Calculate average metrics
    static_ave_iou = statistics.mean(static_ave_iou)
    dynamic_ave_iou = statistics.mean(dynamic_ave_iou)
    lane_ave_iou = statistics.mean(lane_ave_iou)
    average_fps = statistics.mean(fps_list) if fps_list else 0

    # Print results
    print(f"Static Average IOU: {static_ave_iou}")
    print(f"Dynamic Average IOU: {dynamic_ave_iou}")
    print(f"Lane Average IOU: {lane_ave_iou}")
    print(f"Average FPS: {average_fps:.2f}")


if __name__ == "__main__":
    main()
