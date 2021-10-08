import subprocess

import pandas as pd
from tqdm import tqdm

from pathlib import Path

from argparse import ArgumentParser
from pycode2seq.inference.model.model import Model as Code2Seq
from shutil import rmtree

def call_astminer(path, output_path, cli_path):
    subprocess.call(
        f"{cli_path} code2vec --lang java --project {path} --output {Path(output_path, path.name)} --split-tokens --granularity method --hide-method-name",
        cwd=Path(cli_path).parent,
        shell=True)

def predict(data_path: Path, output_path: Path, cli_path: Path):
    folders = list(data_path.glob("*/"))
    model = Code2Seq.load("java")
    for folder in tqdm(folders):
        (output_path / folder.name).mkdir(exist_ok=True, parents=True)
        for filename in Path(folder).glob("*.java"):
            call_astminer(filename, output_path, cli_path)
            if (output_path / filename.name / "java" / "paths.csv").stat().st_size < 10:
                print(f"No paths mined for {filename}")
            else:
                try:
                    embeddigns = model.astminer_embeddings_for_file(output_path / filename.name / "java", "java")
                    df = pd.DataFrame([(name, emb.tolist()) for name, emb in embeddigns], columns=["method", "embedding"])
                    df.to_csv(output_path / folder.name / filename.with_suffix(".csv").name) 
                except Exception as e:
                    print(f"{e} ocurred while processing {filename}")
            rmtree(output_path / filename.name)
            


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("data", type=str)
    arg_parser.add_argument("output", type=str)
    arg_parser.add_argument("cli_path", type=str, help="path to astminer cli.sh")

    args = arg_parser.parse_args()

    predict(Path(args.data), Path(args.output), Path(args.cli_path))