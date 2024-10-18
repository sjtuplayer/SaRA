import subprocess
import csv
import re
import argparse
import json
def run_fidelity_command(command):
    # subprocess
    print_command=command_str = ' '.join(command)
    print(print_command)
    result = subprocess.run(command, capture_output=True, text=True)
    return result.stdout

def parse_clip_score_output(output):
    # Get Text similarity
    clip_score_match = re.search(r'Text similarity: ([\d.]+)', output)
    if clip_score_match:
        clip_score_value = float(clip_score_match.group(1))
        return clip_score_value
    return None
def parse_fid_output(output):
    # Get FID
    fid_match = re.search(r'frechet_inception_distance: ([\d.]+)', output)
    if fid_match:
        fid_value = float(fid_match.group(1))
        return fid_value
    return None
def parse_is_output(output):
    # Get IS
    is_match = re.search(r'inception_score_mean: ([\d.]+)', output)
    if is_match:
        is_value = float(is_match.group(1))
        return is_value
    return None
def write_to_csv(file_path, data):
    # 将数据写入 CSV 文件
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data)
def is_str_in_csv(file_path, target_str):
    try:
        with open(file_path, mode='r', encoding='utf-8') as csvfile:
            csvreader = csv.reader(csvfile)
            for row in csvreader:
                if target_str in row:
                    return True
        return False
    except FileNotFoundError:
        print(f"File {file_path} Not exists")
        return False
    except Exception as e:
        print(f"Reading file erroe: {e}")
        return False
def load_config(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--config",
        type=str,
    )
    parser.add_argument(
        "--target_dir",
        type=str,
    )
    parser.add_argument(
        "--log_path",
        default='log.csv',
        type=str,
        help='path to the log csv file'
    )
    args = parser.parse_args()

    args = parser.parse_args()
    dataset_config = load_config(args.config)
    prefix_name=dataset_config['prefix_name']
    dataset_path='../examples/'+dataset_config['dataset_name']

    gpu_id='0'

    input=args.target_dir
    clip_score_command = [
        'python3', 'clip_score.py', '--target_folder', input, '--prefix_name', f'\"{prefix_name}\"'
    ]
    clip_output = run_fidelity_command(clip_score_command)

    clip_value = parse_clip_score_output(clip_output)

    command = [
        'python3', '-m', 'torch_fidelity.fidelity', '--gpu', gpu_id, '--fid',
        '--input1', dataset_path,
        '--input2', input
    ]

    output = run_fidelity_command(command)

    fid_value = parse_fid_output(output)

    if fid_value is not None :
        data = [args.target_dir,clip_value,fid_value]
        write_to_csv(args.log_path, data)
        print(f"Clip score {clip_value} and FID  {fid_value} have been recorded in {args.log_path} ")
    else:
        print(f"No detexted FID")




