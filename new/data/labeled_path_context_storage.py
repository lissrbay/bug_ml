import shutil
import pickle
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path
from random import shuffle
from typing import Tuple, Optional, List, Dict

import code2seq
from code2seq.data.path_context import LabeledPathContext
from code2seq.data.path_context_dataset import PathContextDataset
from code2seq.data.vocabulary import Vocabulary
from omegaconf import DictConfig
from tqdm import tqdm
from yaml import CLoader as Loader, CDumper as Dumper
from yaml import load, dump

from new.data.report import Frame, Report
from new.model.report_encoders.utils import split_into_subtokens

CACHE_SIZE = 3500


class LabeledPathContextStorage:
    def __init__(self, cli_path: str, config_path: str, vocabulary: Vocabulary, config: DictConfig):
        self.cli_path = cli_path
        self.config_path = config_path
        self.vocabulary = vocabulary
        self.config = config

        self.code_to_filename = {}
        self.mined_contexts = {}

        self.work_dir = Path("/home/dumtrii/Documents/practos/spring2/bug_ml/tmp/storage")
        self.src_path = self.work_dir / "src"
        self.output_path = self.work_dir / "output"
        self.contexts_path = self.work_dir / "contexts"
        self.path_file_path = self.output_path / "java" / "data" / "path_contexts.c2s"

        # improves work time by 10 times ???
        self.max_token_parts = self.config.data.max_token_parts
        self.path_length = self.config.data.path_length
        self.max_context = self.config.data.max_context
        self.max_label_parts = self.config.data.max_label_parts

    def _add_line(self, line: str):
        filename, raw_label, *raw_contexts = line.split()

        if filename in self.mined_contexts and raw_label in self.mined_contexts[filename]:
            return

        processed_line = self._process_line(raw_label, raw_contexts)

        if not processed_line:
            return

        method_name, context = processed_line

        if filename not in self.mined_contexts:
            self.mined_contexts[filename] = {}

        self.mined_contexts[filename][method_name] = context

    def dump_contexts(self):
        for filename in self.mined_contexts:
            file_path = self.contexts_path / filename
            if file_path.exists():
                with open(file_path, "rb") as f:
                    loaded = pickle.load(f)
                self.mined_contexts[filename].update(loaded)

            with open(file_path, "wb") as f:
                pickle.dump(self.mined_contexts[filename], f)

        self.mined_contexts.clear()

    def process_mined_paths(self):
        num_lines = sum(1 for _ in open(self.path_file_path))

        dump_every = 50000

        self.mined_contexts = {}
        with open(self.path_file_path, "r") as f:
            for i, line in tqdm(enumerate(f), total=num_lines):
                self._add_line(line)

                if (i + 1) % dump_every == 0:
                    self.dump_contexts()
            self.dump_contexts()

    def mine_file_contexts(self, file_path: Path) -> Path:
        output_path = self.work_dir / "output"

        with open(self.config_path, "r") as f:
            config = load(f, Loader=Loader)

        config["inputDir"] = str(file_path.parent)
        config["outputDir"] = str(output_path)

        with open(self.config_path, "w") as f:
            dump(config, f, Dumper=Dumper)

        subprocess.call(
            f"{self.cli_path} {self.config_path}",
            cwd=Path(self.cli_path).parent,
            shell=True
        )

        return output_path

    def mine_files(self):
        with open(self.config_path, "r") as f:
            config = load(f, Loader=Loader)

        config["inputDir"] = str(self.src_path)
        config["outputDir"] = str(self.output_path)

        with open(self.config_path, "w") as f:
            dump(config, f, Dumper=Dumper)

        subprocess.call(
            f"{self.cli_path} {self.config_path}",
            cwd=Path(self.cli_path).parent,
            shell=True
        )

    def write_code(self, reports: List[Report]):
        written_files = set()
        file_idx = 0
        for report in tqdm(reports):
            for frame in report.frames:
                code = frame.get_code_decoded()

                if code not in written_files:
                    file_path = self.src_path / f"{file_idx}.java"
                    with open(file_path, "w") as f:
                        f.write(code)
                    self.code_to_filename[frame.code] = str(file_idx)
                    file_idx += 1
                    written_files.add(code)

    def _process_line(self, raw_label: str, raw_contexts: List[str]) -> Optional[Tuple[str, LabeledPathContext]]:
        n_contexts = min(len(raw_contexts), self.max_context)

        shuffle(raw_contexts)

        raw_path_contexts = raw_contexts[:n_contexts]

        # Tokenize label
        label = PathContextDataset.tokenize_label(raw_label, self.vocabulary.label_to_id,
                                                  self.max_label_parts)

        # Tokenize paths
        try:
            paths = [self._get_path(raw_path.split(",")) for raw_path in raw_path_contexts]
            if not paths:
                return None
        except ValueError as e:
            return None

        return raw_label, LabeledPathContext(label, paths)

    def _get_path(self, raw_path: List[str]) -> code2seq.data.path_context.Path:
        return code2seq.data.path_context.Path(
            from_token=PathContextDataset.tokenize_token(raw_path[0], self.vocabulary.token_to_id,
                                                         self.max_token_parts),
            path_node=PathContextDataset.tokenize_token(raw_path[1], self.vocabulary.node_to_id,
                                                        self.path_length),
            to_token=PathContextDataset.tokenize_token(raw_path[2], self.vocabulary.token_to_id,
                                                       self.max_token_parts),
        )

    @lru_cache(maxsize=CACHE_SIZE)
    def get_contexts(self, filename: str) -> Optional[Dict[str, LabeledPathContext]]:
        file_path = self.contexts_path / filename

        if not file_path.exists():
            return None

        with open(file_path, "rb") as f:
            return pickle.load(f)

    def get_frame_contexts(self, frame: Frame) -> Optional[LabeledPathContext]:
        if frame.code not in self.code_to_filename:
            return None

        filename = self.code_to_filename[frame.code]
        method_name = frame.meta["method_name"]

        method_name = self.normalize_method(method_name)

        contexts = self.get_contexts(filename)

        if not contexts:
            return None

        if method_name not in contexts:
            return None

        return contexts[method_name]

    def load_data(self, reports: List[Report], mine_files=True, process_mined=True, remove_all=True):
        # Run with remove_all=False and after that use mine_files=False, process_mined=True
        # to reuse mined data

        self.output_path.mkdir(parents=True, exist_ok=True)
        self.src_path.mkdir(parents=True, exist_ok=True)
        self.contexts_path.mkdir(parents=True, exist_ok=True)

        self.write_code(reports)
        if mine_files:
            self.mine_files()
        if process_mined:
            self.process_mined_paths()

        if remove_all:
            shutil.rmtree(self.output_path)
            shutil.rmtree(self.src_path)

    def normalize_method(self, method: str, separator: str = "|") -> str:
        method_name = method.split(".")[-1]
        return separator.join(split_into_subtokens(method_name)[:self.max_label_parts])
