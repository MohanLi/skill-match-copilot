"""WorkRB skill-extraction ranking tasks that load the benchmark from the LOCAL CSVs (GitHub copy)
instead of Hugging Face (unreachable from the workspace). The grouping/filtering logic mirrors
workrb.tasks.ranking.skill_extraction.BaseESCOSkillExtractRanking.load_dataset exactly, so metrics are
comparable with published WorkRB numbers (task names are unchanged).

ESCO itself comes from src.data.esco.ensure_workrb_esco() — the official CSVs when present, else a
stand-in built from the benchmark's label list (results flagged provisional)."""
from __future__ import annotations

import pandas as pd
from workrb.data.esco import ESCO
from workrb.tasks.abstract.ranking_base import RankingDataset
from workrb.tasks.ranking.skill_extraction import HouseSkillExtractRanking, TechSkillExtractRanking, TechWolfSkillExtractRanking
from workrb.types import DatasetSplit, Language

from ..data import esco as esco_data
from ..data import esco_benchmark as bm

_SUBSET = {"Skill Extraction Tech": "tech", "Skill Extraction House": "house", "Skill Extraction TechWolf": "techwolf"}


class _LocalCSVMixin:
    subset: str = ""

    def load_dataset(self, dataset_id: str, split: DatasetSplit) -> RankingDataset:  # type: ignore[override]
        language = Language(dataset_id)
        assert language == Language.EN, "only English is available locally"
        esco_data.ensure_workrb_esco(self.esco_version, "en")
        split_name = {DatasetSplit.TEST: "test", DatasetSplit.VAL: "validation"}[split]
        if (self.subset, split_name) not in bm.FILES:
            raise ValueError(f"no local file for subset={self.subset} split={split_name}")
        df = bm.load(self.subset, split_name)[["sentence", "label"]]
        grouped = df.groupby("sentence")["label"].apply(list).reset_index()
        esco = ESCO(version=self.esco_version, language=language)
        skill_vocab = esco.get_skills_vocabulary()
        skill2label = {s: i for i, s in enumerate(skill_vocab)}
        queries, labels = [], []
        for q, lst in zip(grouped["sentence"], grouped["label"], strict=True):
            kept = [s for s in lst if s in skill2label]
            if not kept:
                continue
            queries.append(q)
            labels.append([skill2label[s] for s in kept])
        return RankingDataset(query_texts=queries, target_indices=labels, target_space=skill_vocab, dataset_id=dataset_id)


class LocalTechSkillExtractRanking(_LocalCSVMixin, TechSkillExtractRanking):
    subset = "tech"


class LocalHouseSkillExtractRanking(_LocalCSVMixin, HouseSkillExtractRanking):
    subset = "house"


class LocalTechWolfSkillExtractRanking(_LocalCSVMixin, TechWolfSkillExtractRanking):
    subset = "techwolf"


def make_tasks(split: str = "val", subsets: tuple[str, ...] = ("tech", "house", "techwolf"), esco_version: str = "1.1.0") -> list:
    """split='val' has no TECHWOLF file locally, so it is skipped there automatically."""
    classes = {"tech": LocalTechSkillExtractRanking, "house": LocalHouseSkillExtractRanking, "techwolf": LocalTechWolfSkillExtractRanking}
    tasks = []
    for s in subsets:
        if split == "val" and s == "techwolf":
            continue
        tasks.append(classes[s](esco_version=esco_version, split=split, languages=["en"]))
    return tasks


def dataset_sizes(tasks: list) -> dict[str, int]:
    return {t.name: len(t.datasets["en"].query_texts) for t in tasks}
