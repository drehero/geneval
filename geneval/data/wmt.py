import pathlib
import requests
import shutil

import pandas as pd
from tqdm.auto import tqdm


def fetch(url, path):
    fn = url.split("/")[-1]
    r = requests.get(url, stream=True)
    chunk_size = 1024
    pbar = tqdm(
        desc=f"Downloading {fn}",
        unit="B",
        unit_scale=True,
        unit_divisor=chunk_size,
        total=int(r.headers["Content-Length"])
    )
    with open(path, "wb") as f:
        for chunk in r.iter_content(chunk_size=chunk_size):
            if chunk:
                pbar.update(len(chunk))
                f.write(chunk)


class WMT18:
    urls = [
        "http://ufallab.ms.mff.cuni.cz/~bojar/wmt18-metrics-task-package.tgz",
        "http://ufallab.ms.mff.cuni.cz/~bojar/wmt18/wmt18-metrics-task-nohybrids.tgz"
    ]

    def __init__(self, lang_pair, root="./", download=True):
        self.lang_pair = lang_pair
        self.root = pathlib.Path(root)
        self.download = download

        # download and unpack if not exists
        for url in self.urls:
            fn = pathlib.Path(url.split("/")[-1])
            if not (self.root / fn.stem).is_dir():
                if not (self.root / fn).is_file():
                    if self.download:
                        self.root.mkdir(parents=True, exist_ok=True)
                        fetch(url, self.root / fn)
                    else:
                        raise Exception(
                            f"WMT18 files do not exist. Use download=True to download."
                        )
                shutil.unpack_archive(self.root / fn, self.root)

        # load segment level relative ranking (RR-seglevel.csv)
        # https://github.com/Tiiiger/bert_score/blob/master/reproduce/get_wmt18_seg_results.py
        seglevel = pd.read_csv(
            self.root / f"wmt18-metrics-task-package/manual-evaluation/RR-seglevel.csv",
            sep=" "
        ).loc[self.lang_pair, :]
        systems = list(set(seglevel["BETTER"]).union(set(seglevel["WORSE"])))
        translations = {}
        for system in systems:
            sys_path = self.root / f"wmt18-metrics-task-nohybrids/system-outputs/newstest2018/{self.lang_pair}/newstest2018.{system}.{self.lang_pair}"
            with open(sys_path, "r", encoding="UTF-8") as f:
                translations[system] = f.read().split("\n")
        src_lang, tgt_lang = self.lang_pair.split("-")
        ref_path = self.root / f"wmt18-metrics-task-nohybrids/references/newstest2018-{src_lang}{tgt_lang}-ref.{tgt_lang}"
        with open(ref_path, "r", encoding="UTF-8") as f:
            ref = f.read().split("\n")
        src_path = self.root / f"wmt18-metrics-task-nohybrids/sources/newstest2018-{src_lang}{tgt_lang}-src.{src_lang}"
        with open(src_path, "r", encoding="UTF-8") as f:
            src = f.read().split("\n")
        self.references = []
        self.translations_better = []
        self.translations_worse = []
        self.sources = []
        for _, row in seglevel.iterrows():
            self.translations_better += [translations[row["BETTER"]][row["SID"]-1]]
            self.translations_worse += [translations[row["WORSE"]][row["SID"]-1]]
            self.references += [ref[row["SID"]-1]]
            self.sources += [src[row["SID"]-1]]


class WMT17:
    url = "http://ufallab.ms.mff.cuni.cz/~bojar/wmt17-metrics-task-package.tgz"

    def __init__(self, lang_pair, root="./", download=True):
        self.lang_pair = lang_pair
        self.root = pathlib.Path(root)
        self.download = download

        # download and unpack if not exists
        fn = pathlib.Path(self.url.split("/")[-1])
        if not (self.root / fn.stem).is_dir():
            if not (self.root / fn).is_file():
                if self.download:
                    fetch(self.url, self.root / fn)
                else:
                    raise Exception(
                        f"WMT17 files do not exist. Use download=True to download."
                    )
            shutil.unpack_archive(self.root / fn, self.root / fn.stem)
            shutil.unpack_archive(
                self.root / fn.stem / f"input/wmt17-metrics-task-no-hybrids.tgz",
                self.root
            )

        # load segment level direct assesment scores (DA-seglevel.csv)
        seglevel = pd.read_csv(
            self.root / f"wmt17-metrics-task-package/manual-evaluation/DA-seglevel.csv",
            sep=" "
        )
        seglevel = seglevel.loc[(seglevel["LP"] == self.lang_pair) & ~(seglevel["SYSTEM"].str.contains(r"\+")), :]
        systems = list(seglevel["SYSTEM"].unique())
        system_outputs = {}
        for system in systems:
            sys_path = self.root / f"wmt17-metrics-task-no-hybrids/wmt17-submitted-data/txt/system-outputs/newstest2017/{self.lang_pair}/newstest2017.{system}.{self.lang_pair}"
            with open(sys_path, "r", encoding="UTF-8") as f:
                system_outputs[system] = f.read().split("\n")
        src_lang, tgt_lang = self.lang_pair.split("-")
        ref_path = self.root / f"wmt17-metrics-task-no-hybrids/wmt17-submitted-data/txt/references/newstest2017-{src_lang}{tgt_lang}-ref.{tgt_lang}"
        with open(ref_path, "r", encoding="UTF-8") as f:
            refs = f.read().split("\n")
        src_path = self.root / f"wmt17-metrics-task-no-hybrids/wmt17-submitted-data/txt/sources/newstest2017-{src_lang}{tgt_lang}-src.{src_lang}"
        with open(src_path, "r", encoding="UTF-8") as f:
            src = f.read().split("\n")

        self.references = []
        self.translations = []
        self.scores = []
        self.sources = []
        for _, row in seglevel.iterrows():
            self.translations += [system_outputs[row["SYSTEM"]][row["SID"]-1]]
            self.references += [refs[row["SID"]-1]]
            self.scores += [row["HUMAN"]]
            self.sources += [src[row["SID"]-1]]

class WMT16:
    url = "https://www.statmt.org/wmt16/metrics-task/wmt2016-seg-metric-dev-5lps.tar.gz"

    def __init__(self, lang_pair, root="./", download=True):
        self.lang_pair = lang_pair
        self.root = pathlib.Path(root)
        self.download = download

        # download and unpack if not exists
        fn = pathlib.Path(self.url.split("/")[-1])
        if not (self.root / fn.with_suffix("").stem).is_dir():
            if not (self.root / fn).is_file():
                if self.download:
                    fetch(self.url, self.root / fn)
                else:
                    raise Exception(
                        f"WMT16 files do not exist. Use download=True to download."
                    )
            shutil.unpack_archive(self.root / fn, self.root)
        
        path = self.root / fn.with_suffix("").stem / self.lang_pair
        ref_path = path / f"newstest2015.reference.{self.lang_pair}"
        src_path = path / f"newstest2015.source.{self.lang_pair}"
        trans_path = path / f"newstest2015.mt-system.{self.lang_pair}"
        score_path = path / f"newstest2015.human.{self.lang_pair}"
        with open(ref_path, "r", encoding="UTF-8") as f:
            self.references = f.read().strip().split("\n")
        with open(src_path, "r", encoding="UTF-8") as f:
            self.sources = f.read().strip().split("\n")
        with open(trans_path, "r", encoding="UTF-8") as f:
            self.translations = f.read().strip().split("\n")
        with open(score_path, "r", encoding="UTF-8") as f:
            self.scores = list(map(float, f.read().strip().split("\n")))
