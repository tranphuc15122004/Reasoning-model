import os
import pandas as pd
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import glob
import argparse
import sys

def list_event_files(log_dir):
    """Liệt kê tất cả các file event trong thư mục log"""
    event_files = glob.glob(os.path.join(log_dir, "events.out.tfevents.*"))
    if not event_files:
        for root, dirs, files in os.walk(log_dir):
            for file in files:
                if file.startswith("events.out.tfevents."):
                    event_files.append(os.path.join(root, file))
    
    return sorted(event_files)

def extract_tensorboard_data_from_file(event_file):
    """Trích xuất dữ liệu từ một file event TensorBoard cụ thể"""
    event_acc = EventAccumulator(event_file)
    event_acc.Reload()
    
    if 'scalars' not in event_acc.Tags():
        print(f"Không có scalar tags trong file {os.path.basename(event_file)}")
        return None
    
    tags = event_acc.Tags()['scalars']
    if not tags:
        print(f"Không có scalar tags trong file {os.path.basename(event_file)}")
        return None
    
    data_dict = {}
    
    unique_steps = set()
    for tag in tags:
        events = event_acc.Scalars(tag)
        for event in events:
            unique_steps.add(event.step)
    
    steps_list = sorted(list(unique_steps))
    data_dict['step'] = steps_list
    
    for tag in tags:
        metric_name = tag.split('/')[-1]
        events = event_acc.Scalars(tag)
        
        temp_dict = {}
        for event in events:
            temp_dict[event.step] = event.value
        
        values = []
        for step in steps_list:
            values.append(temp_dict.get(step, np.nan))  
        
        data_dict[metric_name] = values
    
    return pd.DataFrame(data_dict)

def plot_metrics(df, output_dir, metrics_to_plot=None, max_rows=None, figsize=(10, 6), dpi=100):
    """
    Vẽ biểu đồ riêng biệt cho từng metric (mỗi metric một file ảnh)
    
    Args:
        df: DataFrame chứa dữ liệu (phải có cột 'step')
        output_dir: Thư mục lưu các biểu đồ
        metrics_to_plot: Danh sách các metrics cụ thể để vẽ (None = tất cả)
        max_rows: Số lượng dòng tối đa để plot (None = tất cả)
        figsize: Kích thước của từng figure (width, height)
        dpi: Độ phân giải của hình
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if max_rows is not None and len(df) > max_rows:
        df = df.head(max_rows)
        print(f"Chỉ sử dụng {max_rows} dòng đầu tiên để vẽ biểu đồ")
    
    available_metrics = [col for col in df.columns if col != 'step']
    
    if metrics_to_plot:
        metrics = [m for m in metrics_to_plot if m in available_metrics]
    else:
        metrics = available_metrics
    
    plot_paths = []
    
    for metric in metrics:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        
        # Vẽ dữ liệu
        ax.plot(df['step'], df[metric], marker='o', linestyle='-', color='royalblue')
        ax.set_title(f'{metric}')
        ax.set_xlabel('Step')
        ax.set_ylabel('Value')
        ax.grid(True)
        fig.tight_layout()
        
        # Lưu biểu đồ
        plot_path = os.path.join(output_dir, f'{metric}_plot.png')
        fig.savefig(plot_path, dpi=dpi)
        plt.close(fig)
        plot_paths.append(plot_path)
    
    print(f"Đã lưu {len(plot_paths)} biểu đồ vào {output_dir}")
    for path in plot_paths:
        print(f"  - {os.path.basename(path)}")
    
    return plot_paths

def process_specific_event_file(event_file, output_dir='output', metrics_to_plot=None, max_rows=None):
    """Xử lý một file event cụ thể"""
    if not os.path.exists(event_file):
        print(f"File {event_file} không tồn tại")
        return
    
    print(f"Đang xử lý file: {os.path.basename(event_file)}")
    
    # Tạo thư mục đầu ra
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.basename(event_file).replace("events.out.tfevents.", "")
    csv_file = os.path.join(output_dir, f"metrics_{base_name}.csv")
    
    df = extract_tensorboard_data_from_file(event_file)
    
    if df is not None and not df.empty:
        df.to_csv(csv_file, index=False)
        print(f"Đã lưu dữ liệu vào {csv_file}")
        
        plot_metrics(df, output_dir, metrics_to_plot=metrics_to_plot, max_rows=max_rows)
    else:
        print(f"Không có dữ liệu để xử lý từ file {os.path.basename(event_file)}")


if __name__ == "__main__":
    event_file = r"loggg\events.out.tfevents.1746630319.1c996045e015"  # Nhập đường dẫn file event vào đây
    output_dir = "output_plots"  # Thư mục đầu ra
    max_rows = 200  # Số dòng tối đa để plot (None = tất cả)
    
    # Danh sách các metrics muốn vẽ (None = tất cả)
    metrics_to_plot = [ 'test_acc1', 'test_acc5', 'test_loss', 'test_f1', 'test_auc', 'test_recall']
    
    # Kiểm tra xem file có tồn tại không
    if os.path.exists(event_file):
        # Xử lý file event
        process_specific_event_file(event_file, output_dir, metrics_to_plot, max_rows)
    else:
        print(f"Lỗi: File {event_file} không tồn tại. Vui lòng kiểm tra đường dẫn.")