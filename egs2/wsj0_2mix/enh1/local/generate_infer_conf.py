import argparse
import yaml
from pathlib import Path

"""
example:
1. re_min and re_max has -1, en_min and en_max don't have -1, generate from en_min to en_max
separator_conf:
    encoder_repeat_times: 4

2. re_min and re_max don't have -1, en_min and en_max don't have -1, generate from re_min to re_max
separator_conf:
    reconstructor_repeat_times: 2

3. re_min and re_max don't have -1, en_min and en_max don't have -1, generate from re_min to re_max
separator_conf:
    encoder_repeat_times: 4
    reconstructor_repeat_times: 2

"""


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf_file", type=str, default="conf/efficient_infer")
    parser.add_argument("--re_min", type=int, default=-1)
    parser.add_argument("--re_max", type=int, default=-1)
    parser.add_argument("--en_min", type=int, default=-1)
    parser.add_argument("--en_max", type=int, default=-1)
    return parser.parse_args()


def generate_conf(conf_file, re_min, re_max, en_min, en_max):
    conf_file = Path(conf_file)
    if re_min == -1 and re_max == -1:
        if en_min == -1 and en_max == -1:
            print("Error: re_min and re_max and en_min and en_max are all -1")
            exit(1)

    if re_min == -1 or re_max == -1:
        if en_min != -1 and en_max != -1 and en_min <= en_max:
            for en in range(en_min, en_max + 1):
                conf = {
                    "separator_conf": {
                        "encoder_repeat_times": en,
                        "encoder_multi_decoder": False,
                        "encoder_n_layers_multi_decoder": False,
                    }
                }
                output_file = conf_file / f"en/en_{en}.yaml"
                output_file.parent.mkdir(parents=True, exist_ok=True)
                with open(output_file, "w") as f:
                    yaml.dump(conf, f)

    if en_min == -1 or en_max == -1:
        if re_min != -1 and re_max != -1 and re_min <= re_max:
            for re in range(re_min, re_max + 1):
                conf = {
                    "separator_conf": {
                        "reconstructor_repeat_times": re,
                        "encoder_decoder": False,
                        "encoder_multi_decoder": False,
                        "encoder_n_layers_multi_decoder": False,
                        "reconstructor_multi_decoder": False,
                        "reconstructor_n_layers_multi_decoder": False,
                        "spliter_loss": False,
                    }
                }
                output_file = conf_file / f"re/re_{re}.yaml"
                output_file.parent.mkdir(parents=True, exist_ok=True)
                with open(output_file, "w") as f:
                    yaml.dump(conf, f)

    if re_min != -1 and re_max != -1 and re_min <= re_max:
        if en_min != -1 and en_max != -1 and en_min <= en_max:
            for en in range(en_min, en_max + 1):
                for re in range(re_min, re_max + 1):
                    conf = {
                        "separator_conf": {
                            "encoder_repeat_times": en,
                            "reconstructor_repeat_times": re,
                            "encoder_decoder": False,
                            "encoder_multi_decoder": False,
                            "encoder_n_layers_multi_decoder": False,
                            "reconstructor_multi_decoder": False,
                            "reconstructor_n_layers_multi_decoder": False,
                            "spliter_loss": False,
                        }
                    }
                    output_file = conf_file / f"en_re/en_{en}_re_{re}.yaml"
                    output_file.parent.mkdir(parents=True, exist_ok=True)
                    with open(output_file, "w") as f:
                        yaml.dump(conf, f)


if __name__ == "__main__":
    args = arg_parse()
    generate_conf(args.conf_file, args.re_min, args.re_max, args.en_min, args.en_max)
    # python local/generate_infer_conf.py --re_min 1 --re_max 8 --en_min 1 --en_max 4
    # python local/generate_infer_conf.py --en_min 1 --en_max 9
    # python local/generate_infer_conf.py --re_min 1 --re_max 6
