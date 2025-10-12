import argparse
import yaml
from pathlib import Path
from typing import Optional
import torch
import sys
import re
import logging
import os
from datetime import datetime
from io import StringIO

from espnet2.bin.enh_inference import SeparateSpeech
from espnet2.torch_utils.device_funcs import to_device
from espnet2.utils.types import str_or_none


class DualLogger:
    """自定义日志类，捕获所有print输出并同时输出到控制台和文件"""
    def __init__(self, log_file_path):
        self.log_file_path = log_file_path
        # 创建日志目录
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        
        # 保存原始的stdout
        self.original_stdout = sys.stdout
        
        # 打开日志文件
        self.log_file = open(log_file_path, 'w', encoding='utf-8')
        
        # 设置重定向
        sys.stdout = self
    
    def write(self, text):
        """重定向sys.stdout的write方法"""
        # 写入到原始stdout（控制台）
        self.original_stdout.write(text)
        self.original_stdout.flush()
        
        # 写入到日志文件，添加时间戳（只对非空行添加）
        if text.strip():
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            self.log_file.write(f"{timestamp} - {text}")
        else:
            self.log_file.write(text)
        self.log_file.flush()
    
    def flush(self):
        """刷新缓冲区"""
        self.original_stdout.flush()
        self.log_file.flush()
    
    def print(self, *args, **kwargs):
        """保持兼容性的print方法"""
        print(*args, **kwargs)
    
    def close(self):
        """恢复原始stdout并关闭日志文件"""
        sys.stdout = self.original_stdout
        self.log_file.close()


def get_enh_exp_dir(enh_config: str, enh_args: Optional[str]) -> Path:
    """
    Infer the enhancement experiment directory path from enhancement config and arguments.
    This function mimics the logic in egs2/TEMPLATE/enh1/enh.sh.
    """
    config_basename = Path(enh_config).stem
    # NOTE(wangyu): feats_type is fixed to raw in enh.sh
    tag = f"{config_basename}_raw"
    if enh_args:
        # Mimic sed commands in shell script
        # sed -e "s/--\|\//\_/g" -e "s/[ |=]//g"
        enh_args_str = re.sub(r"--|\/", "_", enh_args)
        enh_args_str = re.sub(r"[ |=]", "", enh_args_str)
        tag += f"_{enh_args_str}"

    # NOTE(wangyu): expdir is fixed to "exp"
    return Path("exp") / f"enh_{tag}"


def create_log_file_path(enh_config: str, inference_config: Optional[str], profile_type: str) -> str:
    """创建日志文件路径"""
    train_config_name = Path(enh_config).stem
    if inference_config:
        infer_config_name = Path(inference_config).stem
        log_filename = f"{infer_config_name}_{profile_type}.log"
    else:
        log_filename = f"default_{profile_type}.log"
    
    return os.path.join("exp", "profile", train_config_name, log_filename)


def profile(
    enh_exp_dir: Path,
    model_file: str,
    inference_config: Optional[str],
    speech_segment: int,
    speech_length: float,
    fs: int,
    device: str,
    profile_type: str = "calflops",
):
    """
    Profile the enhancement model using calflops.

    Args:
        enh_exp_dir (Path): Path to the enhancement experiment directory.
        model_file (str): Name of the model parameter file.
        inference_config (Optional[str]): Path to the inference configuration file.
        speech_segment (int): Segment length in frames.
        speech_length (float): Speech length in seconds.
        fs (int): Sampling rate.
        device (str): Device to use for inference ("cpu" or "cuda").
    """
    # Using calflops for profiling
    # import according to the profile_type
    if profile_type == "calflops":
        from calflops import calculate_flops
    elif profile_type == "ptflops":
        from ptflops import get_model_complexity_info
    else:
        raise ValueError(f"Invalid profile type: {profile_type}")

    train_config = enh_exp_dir / "config.yaml"
    model_file_path = enh_exp_dir / model_file

    if not train_config.is_file():
        raise FileNotFoundError(f"Training config not found: {train_config}")
    if not model_file_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_file_path}")

    print("Building model...")
    # 1. Build model
    # Use SeparateSpeech to build the model, as it handles all the config loading.
    separate_speech = SeparateSpeech(
        train_config=str(train_config),
        model_file=str(model_file_path),
        normalize_output_wav=True,
        inference_config=inference_config,
        device=device,
    )

    # 2. Prepare dummy input
    if speech_length is None:
        num_samples = int(speech_segment)
        if fs is None:
            # Fallback for older configs that might not have 'fs'
            with (enh_exp_dir / "config.yaml").open("r", encoding="utf-8") as f:
                train_args = yaml.safe_load(f)
            fs = train_args.get("fs", 8000)
        speech_length = speech_segment / fs
    else:
        num_samples = int(speech_length * fs)
    print(
        f"Input audio length: {speech_length}s, Sampling rate: {fs}Hz, Samples: {num_samples}"
    )

    # 3. Profile using calflops
    print("Using calflops for FLOPS calculation.")
    class Wrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model.enh_model
            self.infer_model = model
        def forward(self, x):
            return self.infer_model(x, fs)
    # Use the enhancement model directly for profiling
    infer_model = Wrapper(separate_speech)
    with torch.no_grad():
        if profile_type == "calflops":
            flops, macs, params = calculate_flops(
                model=infer_model,
                input_shape=(1, num_samples),
            )
        elif profile_type == "ptflops":
            def prepare_input(input_shape):
                return torch.randn(input_shape)

            macs, params = get_model_complexity_info(
                infer_model,
                (1, num_samples,),
                input_constructor=prepare_input,
                as_strings=False,
                print_per_layer_stat=True,
                verbose=False
            )
        else:
            raise ValueError(f"Invalid profile type: {profile_type}")

    
    if profile_type == "calflops":
        print(f"\nSummary of {profile_type}:")
        print(f"  Computational complexity: {flops}")
        print(f"  Number of multiply-adds: {macs}")
        print(f"  Number of parameters: {params}")
    elif profile_type == "ptflops":
        print(f"\nSummary of {profile_type}:")
        print(f"  Number of multiply-adds: {macs}")
        print(f"  Number of parameters: {params}")
    else:
        raise ValueError(f"Invalid profile type: {profile_type}")


def main():
    """Main function to run the profiling."""
    parser = argparse.ArgumentParser(
        description="Profile an enhancement model with calflops.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--enh_config",
        type=str,
        required=True,
        help="Source training configuration file (e.g., conf/train.yaml). "
        "Used to infer the experiment directory.",
    )
    parser.add_argument(
        "--enh_args",
        type=str,
        default=None,
        help='Arguments for enhancement model training, e.g., "--max_epoch 10". '
        "Used to infer the experiment directory.",
    )
    parser.add_argument(
        "--model_file",
        type=str,
        default="valid.loss.ave.pth",
        help="Model parameter file name inside the experiment directory.",
    )
    parser.add_argument(
        "--inference_config",
        type=str_or_none,
        default=None,
        help="Optional configuration file for overwriting enh model attributes "
        "during inference.",
    )
    parser.add_argument(
        "--speech_segment",
        type=int,
        default=None,
        help="Segment length in frames. If not provided, it will be read from the config file.",
    )
    parser.add_argument(
        "--speech_length",
        type=float,
        default=None,
        help="Speech length in seconds. If not provided, speech_segment will be used.",
    )
    parser.add_argument(
        "--fs",
        type=int,
        default=None,
        help="Sampling rate. If not provided, speech_segment will be used.",
    )
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device to use for profiling.")
    parser.add_argument("--profile_type", type=str, default="ptflops", choices=["calflops", "ptflops"], help="Type of profile to use.")
    args = parser.parse_args()

    # 创建日志文件路径
    log_file_path = create_log_file_path(args.enh_config, args.inference_config, args.profile_type)
    
    # 初始化日志记录器
    logger = DualLogger(log_file_path)
    
    try:
        print(f"Profile started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Log file: {log_file_path}")
        print("="*60)

        enh_exp_dir = get_enh_exp_dir(args.enh_config, args.enh_args)
        print(f"Inferred experiment directory: {enh_exp_dir}")
        train_config_path = enh_exp_dir / "config.yaml"

        if not train_config_path.is_file():
            raise FileNotFoundError(
                f"Training config 'config.yaml' not found in experiment directory: {enh_exp_dir}\n"
                f"Please make sure --enh_config '{args.enh_config}' and --enh_args '{args.enh_args}' "
                "correctly point to a trained model."
            )

        if args.inference_config and not Path(args.inference_config).is_file():
            raise FileNotFoundError(f"Inference config file not found: {args.inference_config}")

        with train_config_path.open("r", encoding="utf-8") as f:
            train_args = yaml.safe_load(f)

        speech_segment = args.speech_segment
        if speech_segment is None:
            speech_segment = train_args.get("speech_segment")
            if speech_segment is None:
                raise ValueError(
                    "`speech_segment` is not specified via arguments or in the training config."
                )

        fs = args.fs
        if fs is None:
            fs = train_args.get("fs")

        profile(
            enh_exp_dir,
            args.model_file,
            args.inference_config,
            speech_segment,
            args.speech_length,
            fs,
            args.device,
            args.profile_type,
        )
        
        print("="*60)
        print(f"Profile completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        print(f"Error occurred: {e}")
        raise
    finally:
        # 确保恢复原始stdout并关闭日志文件
        logger.close()

def summary():
    """总结profile结果到markdown文件
       Markdown输出:
        | Model | Inference Config | Profile Type | FLOPS | MACs | Params |
        | ----- | ---------------- | ------------ | ----- | ---- | ------ |
        | model1 | config1 | calflops | 1000 | 1000 | 1000 |
        | model1 | config1 | ptflops | - | 2000 | 2000 |
        | model2 | config2 | calflops | 3000 | 3000 | 3000 |
    """
    import glob
    
    profile_dir = "exp/profile"
    if not os.path.exists(profile_dir):
        print(f"Profile目录不存在: {profile_dir}")
        return
    
    results = []
    
    # 1. 递归遍历所有日志文件
    log_files = glob.glob(os.path.join(profile_dir, "**", "*.log"), recursive=True)
    
    for log_file in log_files:
        try:
            result = parse_log_file(log_file)
            if result:
                results.append(result)
        except Exception as e:
            print(f"解析日志文件失败 {log_file}: {e}")
    
    # 2. 按模型名和推理配置排序
    results.sort(key=lambda x: (x['model'], x['inference_config'], x['profile_type']))
    
    # 3. 生成markdown表格
    if not results:
        print("未找到有效的profile结果")
        return
    
    # 创建markdown内容
    markdown_content = generate_markdown_table(results)
    
    # 保存到文件
    output_file = os.path.join(profile_dir, "profile_summary.md")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    
    print(f"Profile总结已保存到: {output_file}")
    print("\n" + markdown_content)


def parse_log_file(log_file):
    """解析单个日志文件，提取profile信息"""
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 提取基本信息
        model_name = extract_model_name(log_file)
        inference_config = extract_inference_config(log_file)
        profile_type = extract_profile_type(log_file)
        
        # 提取性能指标
        if profile_type == "calflops":
            flops, macs, params = extract_calflops_metrics(content)
        elif profile_type == "ptflops":
            flops, macs, params = extract_ptflops_metrics(content)
        else:
            return None
        
        return {
            'model': model_name,
            'inference_config': inference_config,
            'profile_type': profile_type,
            'flops': flops,
            'macs': macs,
            'params': params,
            'log_file': log_file
        }
    except Exception as e:
        print(f"解析日志文件时出错 {log_file}: {e}")
        return None


def extract_model_name(log_file):
    """从日志文件路径提取模型名"""
    # 从路径中提取模型目录名
    # 例如: exp/profile/train_enh_tflocoformer_efficient_residual_en1x6_l1x6/en_6_ptflops.log
    # 提取: train_enh_tflocoformer_efficient_residual_en1x6_l1x6
    parts = log_file.split(os.sep)
    if len(parts) >= 3:
        return parts[-2]  # 倒数第二个部分是模型目录名
    return "unknown"


def extract_inference_config(log_file):
    """从日志文件名提取推理配置名"""
    # 从文件名中提取配置名
    # 例如: en_1_re_1_ptflops.log -> en_1_re_1
    filename = os.path.basename(log_file)
    if filename.endswith('_calflops.log'):
        return filename[:-12]  # 移除'_calflops.log'
    elif filename.endswith('_ptflops.log'):
        return filename[:-11]   # 移除'_ptflops.log'
    else:
        return filename.replace('.log', '')


def extract_profile_type(log_file):
    """从日志文件名提取profile类型"""
    filename = os.path.basename(log_file)
    if 'calflops' in filename:
        return 'calflops'
    elif 'ptflops' in filename:
        return 'ptflops'
    return 'unknown'


def extract_calflops_metrics(content):
    """从calflops日志内容中提取性能指标"""
    flops = None
    macs = None
    params = None
    
    # 查找Summary of calflops部分
    lines = content.split('\n')
    in_summary = False
    
    for line in lines:
        if 'Summary of calflops:' in line:
            in_summary = True
            continue
        
        if in_summary and line.strip():
            # 移除时间戳
            clean_line = re.sub(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} - ', '', line)
            
            if 'Computational complexity:' in clean_line:
                flops = clean_line.split('Computational complexity:')[1].strip()
            elif 'Number of multiply-adds:' in clean_line:
                macs = clean_line.split('Number of multiply-adds:')[1].strip()
            elif 'Number of parameters:' in clean_line:
                params = clean_line.split('Number of parameters:')[1].strip()
        
        # 结束summary部分
        if in_summary and '====' in line:
            break
    
    return flops, macs, params


def extract_ptflops_metrics(content):
    """从ptflops日志内容中提取性能指标"""
    flops = None  # ptflops不提供FLOPS
    macs = None
    params = None
    
    # 查找Summary of ptflops部分
    lines = content.split('\n')
    in_summary = False
    
    for line in lines:
        if 'Summary of ptflops:' in line:
            in_summary = True
            continue
        
        if in_summary and line.strip():
            # 移除时间戳
            clean_line = re.sub(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} - ', '', line)
            
            if 'Number of multiply-adds:' in clean_line:
                macs_raw = clean_line.split('Number of multiply-adds:')[1].strip()
                # 转换为更易读的格式
                try:
                    macs_num = int(macs_raw)
                    if macs_num >= 1e9:
                        macs = f"{macs_num/1e9:.2f} GMac"
                    elif macs_num >= 1e6:
                        macs = f"{macs_num/1e6:.2f} MMac"
                    elif macs_num >= 1e3:
                        macs = f"{macs_num/1e3:.2f} KMac"
                    else:
                        macs = f"{macs_num} Mac"
                except:
                    macs = macs_raw
            elif 'Number of parameters:' in clean_line:
                params_raw = clean_line.split('Number of parameters:')[1].strip()
                # 转换为更易读的格式
                try:
                    params_num = int(params_raw)
                    if params_num >= 1e6:
                        params = f"{params_num/1e6:.2f} M"
                    elif params_num >= 1e3:
                        params = f"{params_num/1e3:.2f} K"
                    else:
                        params = f"{params_num}"
                except:
                    params = params_raw
        
        # 结束summary部分
        if in_summary and '====' in line:
            break
    
    return flops, macs, params


def generate_markdown_table(results):
    """生成markdown格式的表格"""
    markdown = []
    markdown.append("# Profile Results Summary")
    markdown.append("")
    markdown.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    markdown.append("")
    markdown.append("| Model | Inference Config | Profile Type | FLOPS | MACs | Params |")
    markdown.append("| ----- | ---------------- | ------------ | ----- | ---- | ------ |")
    
    for result in results:
        model = result['model']
        config = result['inference_config']
        profile_type = result['profile_type']
        flops = result['flops'] or '-'
        macs = result['macs'] or '-'
        params = result['params'] or '-'
        
        markdown.append(f"| {model} | {config} | {profile_type} | {flops} | {macs} | {params} |")
    
    markdown.append("")
    markdown.append("## 详细信息")
    markdown.append("")
    for i, result in enumerate(results, 1):
        markdown.append(f"{i}. **{result['model']}** - {result['inference_config']} ({result['profile_type']})")
        markdown.append(f"   - 日志文件: `{result['log_file']}`")
        if result['flops']:
            markdown.append(f"   - FLOPS: {result['flops']}")
        markdown.append(f"   - MACs: {result['macs']}")
        markdown.append(f"   - Parameters: {result['params']}")
        markdown.append("")
    
    return '\n'.join(markdown)


def main_summary():
    """独立运行summary功能的主函数""" 
    parser = argparse.ArgumentParser(
        description="总结所有profile结果到markdown文件",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--profile_dir',
        type=str,
        default='exp/profile',
        help='Profile结果目录路径'
    )
    args = parser.parse_args()
    
    # 切换到profile目录并执行summary
    original_dir = os.getcwd()
    try:
        if args.profile_dir != 'exp/profile':
            # 如果指定了自定义目录，需要调整全局变量
            global profile_dir
            profile_dir = args.profile_dir
        summary()
    finally:
        os.chdir(original_dir)


if __name__ == "__main__":
    # 检查是否要执行summary功能
    if len(sys.argv) > 1 and sys.argv[1] == 'summary':
        # 移除'summary'参数，让argparse正常工作
        sys.argv.pop(1)
        main_summary()
    else:
        main()
