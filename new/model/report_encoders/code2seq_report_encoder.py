import torch
import torch.nn as nn
from code2seq.data.path_context import BatchedLabeledPathContext
from code2seq.model import Code2Seq
from commode_utils.training import cut_into_segments

from new.data.labeled_path_context_storage import LabeledPathContextStorage
from new.data.report import Report
from new.model.report_encoders.report_encoder import ReportEncoder


class Code2SeqReportEncoder(ReportEncoder, nn.Module):

    def __init__(self, model: Code2Seq, context_storage: LabeledPathContextStorage, frames_limit: int = 80,
                 batch_size: int = 32, padding_idx: int = 2):
        super().__init__()
        self.model = model
        self.context_storage = context_storage
        self.frames_limit = frames_limit

        self.model.train()
        self.batch_size = batch_size
        self.padding_idx = padding_idx

    def encode_report(self, report: Report) -> torch.Tensor:

        path_contexts = []
        context_indexes = []

        for index, frame in enumerate(report.frames[:self.frames_limit]):
            contexts = self.context_storage.get_frame_contexts(frame)
            if contexts is not None:
                path_contexts.append(contexts)
                context_indexes.append(index)

        final_embeddings = torch.zeros((len(report.frames[:self.frames_limit]), self.dim), device=self.model.device)

        if path_contexts:
            for i in range(0, len(path_contexts), self.batch_size):
                contexts_chunk, indexes_chunk = path_contexts[i:i + self.batch_size], context_indexes[
                                                                                      i:i + self.batch_size]

                batch = BatchedLabeledPathContext(contexts_chunk)
                batch.move_to_device(self.model.device)
                encoder_output = self.model._encoder(batch.from_token, batch.path_nodes, batch.to_token)

                batched_encoder_output, _ = cut_into_segments(encoder_output, batch.contexts_per_label,
                                                              self.model._decoder._negative_value)

                embeddings = batched_encoder_output[:, -1, :]
                final_embeddings[indexes_chunk] = embeddings

        return final_embeddings

    @property
    def dim(self) -> int:
        return self.model.hparams["model_config"]["decoder_size"]
