"""Text-filtering primitives for pretraining-data cleaning.

每个函数对应 PDF 的一个小组件: HTML→文本、语言识别、PII 掩码、
有害内容(NSFW / toxic speech)、Gopher 质量规则。模型文件从
`get_shared_assets_path() / classifiers` 里找(本机 `shared-data/`, 服务器 `/shared-data`)。
"""
import re
from functools import lru_cache
from pathlib import Path

import fasttext
from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding

from lm_data.common import get_shared_assets_path

_CLASSIFIERS_DIR: Path = get_shared_assets_path() / "classifiers"


def extract_text_from_html_bytes(html_bytes: bytes) -> str:
    """把一个 HTML 页面的原始字节转成纯文本。

    流程: 检测编码 → 按编码解码成 str → 用 resiliparse 提取可见文本。
    编码检测让函数对非 UTF-8 网页也健壮。
    """
    encoding = detect_encoding(html_bytes)
    html = html_bytes.decode(encoding)
    return extract_plain_text(html)


# --------------------------------------------------------------------------- #
# §2.3 语言识别
# --------------------------------------------------------------------------- #

@lru_cache(maxsize=None)
def _load_fasttext_model(model_path: Path):
    """每个进程只加载一次 fastText 模型(lid.176.bin 有 131MB, 加载很慢)。"""
    return fasttext.load_model(str(model_path))


def identify_language(text: str) -> tuple[str, float]:
    """识别一段文本的主要语言。

    返回 (语言代码, 置信度)。fastText 返回形如 `__label__en` 的标签,
    这里剥掉 `__label__` 前缀, 使英文返回 "en"、中文返回 "zh"。
    置信度是模型对 top-1 标签给出的概率(理论上 0~1, fastText 的 softmax
    输出可能有 ±1e-5 的误差, 略超 1.0 属正常)。
    """
    model = _load_fasttext_model(_CLASSIFIERS_DIR / "lid.176.bin")
    labels, probs = model.predict(" ".join(text.split()), k=1)
    language = labels[0].removeprefix("__label__")
    return language, float(probs[0])


# --------------------------------------------------------------------------- #
# §2.4 PII 掩码
# --------------------------------------------------------------------------- #

def _mask_with(text: str, pattern: re.Pattern, mask: str, *, validate=None):
    """把 `pattern` 匹配到的内容替换成 `mask`, 返回 (新文本, 替换次数)。

    `validate` 是可选的过滤回调: 返回 False 的匹配(如无效 IP)会原样保留,
    不计入次数。
    """
    count = 0

    def _repl(match):
        nonlocal count
        token = match.group(0)
        if validate is not None and not validate(token):
            return token
        count += 1
        return mask

    return pattern.sub(_repl, text), count


_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
# 美国常见电话号码: 可选国家码 +1, 可选括号区号, 三位区号-三位-四位。
_PHONE_RE = re.compile(r"(?<!\d)(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}(?!\d)")
# IPv4: 4 段 1~3 位数字用点分隔(段值 0~255 用 validate 检查)。
_IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")


def _is_valid_ipv4(token: str) -> bool:
    return all(0 <= int(octet) <= 255 for octet in token.split("."))


def mask_emails(text: str) -> tuple[str, int]:
    """把文本中的邮箱地址替换为 ``|||EMAIL_ADDRESS|||``, 返回 (新文本, 次数)。"""
    return _mask_with(text, _EMAIL_RE, "|||EMAIL_ADDRESS|||")


def mask_phone_numbers(text: str) -> tuple[str, int]:
    """把文本中的(美国常见格式)电话号码替换为 ``|||PHONE_NUMBER|||``。"""
    return _mask_with(text, _PHONE_RE, "|||PHONE_NUMBER|||")


def mask_ips(text: str) -> tuple[str, int]:
    """把文本中的 IPv4 地址替换为 ``|||IP_ADDRESS|||``(校验每段 <=255)。"""
    return _mask_with(text, _IP_RE, "|||IP_ADDRESS|||", validate=_is_valid_ipv4)


# --------------------------------------------------------------------------- #
# §2.5 有害内容检测
# --------------------------------------------------------------------------- #

def classify_nsfw(text: str) -> tuple[str, float]:
    """用 Dolma 的 Jigsaw NSFW 分类器判断文本是否含 NSFW 内容。

    返回 (标签, 置信度), 标签为 "nsfw" 或 "non-nsfw"(剥掉 `__label__` 前缀)。
    """
    model = _load_fasttext_model(_CLASSIFIERS_DIR / "dolma_fasttext_nsfw_jigsaw_model.bin")
    labels, probs = model.predict(" ".join(text.split()), k=1)
    return labels[0].removeprefix("__label__"), float(probs[0])


def classify_toxic_speech(text: str) -> tuple[str, float]:
    """用 Dolma 的 Jigsaw hate-speech 分类器判断文本是否为有毒言论。

    返回 (标签, 置信度), 标签为 "toxic" 或 "non-toxic"。
    """
    model = _load_fasttext_model(_CLASSIFIERS_DIR / "dolma_fasttext_hatespeech_jigsaw_model.bin")
    labels, probs = model.predict(" ".join(text.split()), k=1)
    return labels[0].removeprefix("__label__"), float(probs[0])


# --------------------------------------------------------------------------- #
# §2.7 质量分类器
# --------------------------------------------------------------------------- #

QUALITY_CLASSIFIER_PATH: Path = _CLASSIFIERS_DIR / "quality_classifier.bin"


def classify_quality(text: str) -> tuple[str, float]:
    """用自训练的质量分类器判断文本是"高质(wiki)"还是"低质(cc)"。

    模型由 scripts/train_quality_classifier.py 训练:
    Wikipedia 外链页为正例, Common Crawl 随机页为负例。
    """
    model = _load_fasttext_model(QUALITY_CLASSIFIER_PATH)
    labels, probs = model.predict(" ".join(text.split()), k=1)
    return labels[0].removeprefix("__label__"), float(probs[0])


# --------------------------------------------------------------------------- #
# §2.6 Gopher 质量规则
# --------------------------------------------------------------------------- #

def gopher_quality_filter(text: str) -> bool:
    """Gopher paper 的质量启发式规则, 全部满足才返回 True。

    规则:
      1. 词数在 [50, 100000] 之间;
      2. 平均词长在 [3, 10] 个字符;
      3. 不超过 30% 的行以 "..." 结尾;
      4. >=80% 的词至少含一个字母。
    """
    words = text.split()
    if not 50 <= len(words) <= 100_000:
        return False

    mean_word_len = sum(len(word) for word in words) / len(words)
    if not 3 <= mean_word_len <= 10:
        return False

    lines = text.split("\n")
    n_ellipsis_lines = sum(1 for line in lines if line.rstrip().endswith("..."))
    if n_ellipsis_lines / len(lines) > 0.30:
        return False

    n_alphabetic_words = sum(1 for word in words if any(ch.isalpha() for ch in word))
    if n_alphabetic_words / len(words) < 0.80:
        return False

    return True
