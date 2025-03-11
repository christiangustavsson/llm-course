import re
import random
import datasets

_N_DATA_FILES = 21

# Randomly select 5 files
random.seed(42)  # Ensure reproducibility
selected_files = random.sample(range(_N_DATA_FILES), 5)
_DATA_FILES = ["subsets/urlsf_subset{:02d}.tar".format(i) for i in selected_files]

class Openwebtext(datasets.GeneratorBasedBuilder):
    """The Open WebText dataset."""

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="plain_text",
            description="Plain text",
            version=datasets.Version("1.0.0"),
        )
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features({"text": datasets.Value("string")}),
            homepage="https://skylion007.github.io/OpenWebTextCorpus/",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        archives = dl_manager.download(_DATA_FILES)
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={
                "archive_iterators": [
                    dl_manager.iter_archive(archive) for archive in archives
                ],
                "iter_archive": dl_manager.iter_archive
            }),
        ]

    def _generate_examples(self, archive_iterators, iter_archive):
        """Yields examples."""
        for archive_iterator in archive_iterators:
            for xz_filepath, xz_f in archive_iterator:
                if not xz_filepath.endswith(".xz"):
                    continue
                for txt_filepath, txt_f in iter_archive(xz_f):
                    if not txt_filepath.endswith(".txt"):
                        continue
                    idx = f"{xz_filepath}/{txt_filepath}"
                    yield idx, {"text": re.sub("\n\n\n+", "\n\n", txt_f.read().decode("utf-8")).strip()}
