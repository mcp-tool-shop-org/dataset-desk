"""Small stand-ins for torch, transformers, sentence_transformers and plain_sight.

Injected into sys.modules for one test at a time (mock.patch.dict), so the suite behaves the same on a machine
where the real libraries are installed. Nothing here loads a model or touches the network."""

import contextlib
import sys
import types
from unittest import mock

import numpy as np


class FakeTensor:
    def __init__(self, arr):
        self.arr = np.asarray(arr, dtype=np.float32)

    def __getitem__(self, item):
        return FakeTensor(self.arr[item])

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.arr


class FakeBatch(dict):
    """What a processor returns: a dict that supports .to(device)."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.devices = []

    def to(self, device):
        self.devices.append(device)
        return self


def make_torch(bf16=False, free_mib=32768, total_mib=32768):
    t = types.ModuleType("torch")
    t.calls = []
    t.interop_raises = False
    t.bfloat16, t.float32 = "bfloat16", "float32"
    t.is_tensor = lambda x: isinstance(x, FakeTensor)
    t.set_num_threads = lambda n: t.calls.append(("threads", n))

    def interop(n):
        t.calls.append(("interop", n))
        if t.interop_raises:
            raise RuntimeError("cannot set interop threads after parallel work started")

    t.set_num_interop_threads = interop

    @contextlib.contextmanager
    def no_grad():
        t.calls.append(("no_grad", None))
        yield

    t.no_grad = no_grad
    t.cuda = types.SimpleNamespace(
        is_bf16_supported=lambda: bf16,
        mem_get_info=lambda *a: (int(free_mib) * (1 << 20), int(total_mib) * (1 << 20)),
    )
    return t


class FakeParam:
    def __init__(self, n):
        self.n = n

    def numel(self):
        return self.n


class FakeModel:
    """Stands in for a transformers model (SigLIP2 or Florence-2)."""

    def __init__(self, pooled_as_object=False, config=None):
        self.pooled_as_object = pooled_as_object
        self.config = config
        self.devices, self.evals, self.generate_kwargs = [], 0, []

    def to(self, device):
        self.devices.append(device)
        return self

    def eval(self):
        self.evals += 1
        return self

    def parameters(self):
        return [FakeParam(10), FakeParam(5)]

    def _feat(self, n, base):
        arr = np.tile(np.arange(1, 5, dtype=np.float32) * base, (n, 1)) * 3.0
        t = FakeTensor(arr)
        return types.SimpleNamespace(pooler_output=t) if self.pooled_as_object else t

    def get_image_features(self, **inputs):
        return self._feat(inputs["n"], 1.0)

    def get_text_features(self, **inputs):
        return self._feat(inputs["n"], 2.0)

    def __call__(self, **inputs):
        return self._feat(inputs["n"], 1.0)

    def generate(self, **kw):
        self.generate_kwargs.append(kw)
        return "gen-ids"


class FakeProcessor:
    def __init__(self):
        self.calls = []

    def __call__(self, **kw):
        self.calls.append(kw)
        src = kw.get("images", kw.get("text"))
        n = len(src) if isinstance(src, list) else 1
        return FakeBatch(n=n, **{k: v for k, v in kw.items() if k not in ("images", "text")})

    def batch_decode(self, ids, skip_special_tokens=False):
        return ["<DETAILED_CAPTION>raw text"]

    def post_process_generation(self, text, task, image_size):
        self.post = (text, task, image_size)
        return {task: "  a lantern on a wall  "}


def make_transformers(processor=None, model=None):
    tf = types.ModuleType("transformers")
    tf.calls = []
    tf.processor = processor or FakeProcessor()
    tf.model = model or FakeModel()

    class AutoProcessor:
        @staticmethod
        def from_pretrained(repo, **kw):
            tf.calls.append(("processor", repo, kw))
            return tf.processor

    class AutoModel:
        @staticmethod
        def from_pretrained(repo, **kw):
            tf.calls.append(("model", repo, kw))
            return tf.model

    class AutoModelForImageTextToText(AutoModel):
        pass

    class Florence2ForConditionalGeneration(AutoModel):
        pass

    class AutoImageProcessor:
        @staticmethod
        def from_pretrained(repo, **kw):
            tf.calls.append(("image_processor", repo, kw))
            return tf.processor

    tf.AutoProcessor, tf.AutoModel = AutoProcessor, AutoModel
    tf.AutoImageProcessor = AutoImageProcessor
    tf.AutoModelForImageTextToText = AutoModelForImageTextToText
    tf.Florence2ForConditionalGeneration = Florence2ForConditionalGeneration
    return tf


class FakeSentenceTransformer:
    instances = []

    def __init__(self, repo, **kw):
        self.repo, self.kw = repo, kw
        self.encoded = []
        self.mod0 = types.SimpleNamespace(
            processing_kwargs=None,
            processor=types.SimpleNamespace(image_seq_length=None,
                                            image_processor=types.SimpleNamespace(max_soft_tokens=None)),
        )
        self.empty = False
        self.instances.append(self)

    def __len__(self):
        return 0 if self.empty else 1

    def __getitem__(self, i):
        return self.mod0

    def parameters(self):
        return [FakeParam(7), FakeParam(3)]

    def encode(self, inputs, **kw):
        self.encoded.append((list(inputs), kw))
        arr = np.tile(np.arange(1, 1025, dtype=np.float32), (len(inputs), 1))
        dim = kw.get("truncate_dim") or 4
        return arr[:, :dim]


def make_sentence_transformers():
    st = types.ModuleType("sentence_transformers")
    st.SentenceTransformer = FakeSentenceTransformer
    FakeSentenceTransformer.instances = []
    return st


@contextlib.contextmanager
def fake_modules(**mods):
    """Scoped sys.modules injection; restores the originals (or absence) on exit."""
    with mock.patch.dict(sys.modules, mods):
        yield
