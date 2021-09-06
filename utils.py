# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 12:00:07 2020

@author: Odegov Ilya, Kovtun Nikolay, Obidin Egor
"""
from collections.abc import Iterable, Iterator
from logging import getLogger, FileHandler, Formatter
from logging import DEBUG, INFO
import os
from random import seed, shuffle
import re
from time import sleep
import warnings


from numpy import asarray, hstack, where
from numpy.random import seed as np_seed
from pandas import Series, read_csv
import requests as req
from requests.exceptions import RequestException
from tqdm import tqdm
import torch
from torch import manual_seed, cuda, backends
from torch import tensor, long
from transformers import BertTokenizer
from transformers import BertModel


warnings.filterwarnings("ignore")


# make logger for projectdif
current_dir = os.path.dirname(os.path.realpath(__file__))
logger_path = os.path.join(current_dir, "logs")
os.makedirs(logger_path, exist_ok=True)
logger = getLogger("logger")
logger.setLevel(DEBUG)
logger_path = os.path.join(logger_path, "project.log")
handler = FileHandler(logger_path, mode="w")
handler.setLevel(DEBUG)
FMT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DATEFMT = "%Y-%m-%d %H:%M:%S"
formatter = Formatter(fmt=FMT, datefmt=DATEFMT)
handler.setFormatter(formatter)
logger.addHandler(handler)


HEADERS = {
    "User-Agent": "my web scraping program. contact me at iavode@mail.ru"
}
# dict for translate from translit to russian letter
STAIGHT_CONVERT = {
    'a':'а', 'b':'б', 'd':'д', 'e':'е', 'f':'ф', 'g':'г', 'h':'х', 'c':'ц',
    'i':'и', 'j':'ж', 'k':'к', 'l':'л', 'n':'н', 'm':'м', 'o':'о', 'p':'п',
    'r':'р', 's':'с', 't':'т', 'u':'у', 'v':'в', 'y':'ы', 'z':'з', ' ':' '
}
FROM_TWO_LETER_TO_ONE = {
    'zh':'ж', 'kh':'х', 'ts':'ц', 'ch':'ч', 'sh': 'ш', 'ie':'ъ', 'yu':'ю', 'ya':'я'
}
FROM_FOUR_LETTER_TO_ONE = {'shch':'щ'}


def binary_balance(
        data, target, relation=0.5, output_size=None, shuffle=False,
        random_state=42, reset_index=True):
    """
    Возвращает датафрейм с заданным отношением класса 1 к общей выборке.

    Параметры:
        data - датафейм или путь до csv файла.

        target - название колонки с таргетом.

        relation - требуемая относительная часть класса 1 в общей выборке, \
        остальное занимает класс 0.

        output_size - выходной размер датафрейма.

        shuffle - флаг отвечающий за перемешивание выходного датафрейма.

        reset_index - обновление индексов.

        @author: Egor Obidin.

    """
    if isinstance(data, str):
        data = read_csv(data)

    if shuffle:
        data = data.sample(frac=1, random_state=random_state)

    if relation > 1 or relation < 0:
        return data

    counts = data[target].value_counts()
    sum_elements = counts[0] + counts[1]
    relation_class_1 = counts[1] / (sum_elements)
    index_drop = []
    delta = 0

    if relation_class_1 < relation:
        delta = sum_elements - int(counts[1] / relation)
        index_drop = where(data[target] == 0)[0][-delta:]
        counts[0] -= delta
    if relation_class_1 > relation:
        delta = sum_elements - int(counts[0] / (1 - relation))
        index_drop = where(data[target] == 1)[0][-delta:]
        counts[1] -= delta

    if output_size is not None and sum_elements - delta > output_size:
        size_class_1 = int(output_size * relation)
        len_drop_class_1 = counts[1] - size_class_1
        len_drop_class_0 = counts[0] - (output_size - size_class_1)

        if relation_class_1 < relation:
            index_drop_class_1 = where(data[target] == 1)[0][-len_drop_class_1:]
            index_drop_class_0 = where(
                data[target] == 0)[0][-len_drop_class_0 - delta:]
        else:
            index_drop_class_1 = where(
                data[target] == 1)[0][-len_drop_class_1 - delta:]
            index_drop_class_0 = where(data[target] == 0)[0][-len_drop_class_0:]

        index_drop = hstack((index_drop_class_1, index_drop_class_0))

    data = data.drop(index=data.index[index_drop])

    if reset_index:
        return data.reset_index(drop=True)
    else:
        return data


def cleanup(text):
    """
    Provide text to lower-case, clean from link, puncts, tag and other.

    @author: Kovtun Nikolay

    """
    text = text.lower()
    text = re.sub(re.compile(r"<[^>]*>"), " ", text)
    text = re.sub(re.compile(r"^\[id\d*|.*\],*\s*"), "", text)
    text = re.sub(re.compile(
        r"(&quot;)|(&lt;)|(&gt;)|(&amp;)|(&apos;)"), " ", text
    )
    text = re.sub(re.compile(
        r"https?://(www\.)?[-a-zA-Z0-9@:%._+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_+.~#?&/=]*)"),
        " ", text
    )
    text = re.sub(re.compile(r"\[[^\[\]]+\|([^\[\]]+)\]"), r"\1", text)
    text = re.sub(re.compile(r"(&#\d+;)"), " ", text)
    text = re.sub(re.compile(
        r"[.,!?\-;:)(_#*=^/`@«»©…“•—<>\[\]\"'+%|&]"), " ", text
        )
    text = text.replace("  ", " ").replace("  ", " ").replace("  ", " ")
    return text


def convert_from_transilt(text):
    """Convert input text from translit to russian language.

    Args:
        text (str): text on translit.

    Returns:
        result (str): converted text.

    @author: Kovtun Nikolay
    """
    result = ""
    i = 0
    while i < len(text):
        if i + 4 <= len(text) and text[i:i+4] in FROM_FOUR_LETTER_TO_ONE.keys():
            result += FROM_FOUR_LETTER_TO_ONE[text[i:i+4]]
            i+=3
        elif i + 2 <= len(text) and text[i:i+2] in FROM_TWO_LETER_TO_ONE.keys():
            result += FROM_TWO_LETER_TO_ONE[text[i:i+2]]
            i+=1
        elif text[i] in STAIGHT_CONVERT.keys():
            result += STAIGHT_CONVERT[text[i]]
        i += 1
    return result


def count_parameters(model):
    """Count parametres with requires grad."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_page(url, logger):
    """Get internet page by url and check all errrs and exceptions.

    Args:
        url (str): link for request.

        logger (Logger): specific logger.

    Returns:
        (Response): request result.
    """
    try:
        page = req.get(url, headers=HEADERS)
        return page
    except RequestException as err:
        msg = f"Error was received when requesting the {url}"
        log(logger, INFO, msg)
        log(logger, INFO, err)
        return None


def log(logger, level, msg):
    """Log messages.

    Args:
        logger (Logger): specific logger.

        level : log level.

        msg (str): log message.

    """
    logger.log(level=level, msg=msg)


def set_seed(_seed):
    """Set all seeds to make results reproducible."""
    manual_seed(_seed)
    cuda.manual_seed_all(_seed)
    backends.cudnn.deterministic = True
    backends.cudnn.benchmark = False
    np_seed(_seed)
    seed(_seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def smart_batching(dataset, tokenizer, max_length, batch_size, clean=True):
    """Make smart padding for every batch.

    Use less memory upgrde compute speed.

    Args:
        dataset (DataFrame): dataframe with text for batching.

        tokenizer (BertTokenizer): tokenizer from huggingface.com.

        max_length (int): max quantity of token in single text sequence.

        batch_size (int): batch size

        clean (bool, optional): flag for clean text from tag, utl and etc. Defaults to True.

    Returns:
        (list): list of lists with tokens.

        @author: Odegov Ilya.

    """
    tqdm.pandas()
    text, label = dataset.columns
    preproc_text = PreprocessText(tokenizer, max_length, clean)
    # tokenize
    dataset["tokens"] = dataset[text].progress_apply(preproc_text.preprocess)
    # get token's sequence len
    dataset["tokens len"] = dataset["tokens"].progress_apply(
        lambda tokens: len(tokens)
    )
    # sort by len
    dataset = dataset.sort_values(by="tokens len", ascending=True)
    print(f"Min seq len: {dataset['tokens len'].iloc[0]}.")
    print(f"Average seq len: {dataset['tokens len'].mean()}.")
    print(f"Max seq len: {dataset['tokens len'].iloc[-1]}.")
    sleep(0.25)
    data = dataset[["tokens", label]].values.tolist()
    batchs = []
    # create batch sequence
    while data:
        to_take = min(len(data), batch_size)
        batch = data[0: to_take]
        batchs.append(batch)
        del data[:to_take]
    dataset = []
    for batch in tqdm(batchs, desc="Batchs padding"):
        max_len = len(batch[-1][0])
        padd_batch = {"input_ids": [], "attention_mask": []}
        targets = []
        for tokens, target in batch:
            # quantity of pad token
            num_padds = max_len - len(tokens)
            # create padd token sequence
            padded_input_ids = tokens + [0] * num_padds
            # create attention_mask
            attention_mask = (
                [1] * len(tokens) + [tokenizer.pad_token_id] * num_padds
            )
            # append pad token and attention mask to dict by special key
            padd_batch["input_ids"].append(padded_input_ids)
            padd_batch["attention_mask"].append(attention_mask)
            # update target list
            targets.append(target)
        # convert to tensor
        padd_batch["input_ids"] = tensor(
            padd_batch["input_ids"], dtype=long)
        padd_batch["attention_mask"] = tensor(
            padd_batch["attention_mask"], dtype=long)
        # update final dataset with padding batchs
        dataset.append((padd_batch, tensor(targets, dtype=long)))
    return dataset


class PreprocessText:
    """
    Class for clean text from different garbage and tokenize it.

    @author: Odegov Ilya.

    """

    def __init__(self, tokenizer_or_name_or_path, max_length, clean):
        """Create class objects."""
        tqdm.pandas()
        self.tokenizer = tokenizer_or_name_or_path
        self._max_length = max_length
        self._clean = clean

    def _process_single_string(self, text, return_tensors):
        """Process single string.

        Clean, tokenize.

        Args:
            text (str): single text sequence.

            return_tensors (str): output format. Can be None, pt, tf.

        Returns:
            [list, Tensor]: tokenize sequence.

        """
        if self._clean:
            text = cleanup(text)
        if return_tensors:
            tokens = self.tokenizer.encode(
                text, add_special_tokens=True, max_length=self._max_length,
                truncation=True, return_tensors=return_tensors,
            )
        else:
            tokens = self.tokenizer.encode(
                text, add_special_tokens=True, max_length=self._max_length,
                truncation=True,
            )
        return tokens

    def preprocess(self, text, return_tensors=None):
        """Clean and tokenize.

        Args:
            text (str, Iterable, Iterator, DataFrame): text sequences.

            return_tensors (str, optional): output format. Can be None, pt, tf.
            Defaults to None.

        Raises:
            TypeError: text has wrong type.

        Returns:
            (Iterator): iterator with converting text to tokens in lazy styles.
        """
        if isinstance(text, str):
            return self._process_single_string(text, return_tensors)
        elif isinstance(text, Series):
            # use Series method to apply process functionto singlr string
            tokens = (
                self._process_single_string(string, return_tensors)
                for string in text.values
            )
            return tokens
        elif isinstance(text, (Iterable, Iterator)):
            # all iterable data type
            result = (
                self._process_single_string(string, return_tensors)
                for string in tqdm(text)
            )
            return result
        else:
            raise TypeError((
                "Input text has wrong type. " +
                f"Need 'str' or 'Iterable', get {type(text)}."
            ))

    @property
    def tokenizer(self):
        """Getter for tokenizer."""
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, tokenizer_name_or_path):
        """Set private variable _tokenizer."""
        if isinstance(tokenizer_name_or_path, BertTokenizer):
            # get already define tokenizer
            self._tokenizer = tokenizer_name_or_path
        elif isinstance(tokenizer_name_or_path, str):
            # get path to model or model name from huggingface.co
            self._tokenizer = BertTokenizer.from_pretrained(
                tokenizer_name_or_path, do_lower_case=True,
            )
        elif isinstance(tokenizer_name_or_path, Iterable):
            # get path in separate format [folder, model_name]
            self._tokenizer = BertTokenizer.from_pretrained(
                os.path.join(*tokenizer_name_or_path), do_lower_case=True,
            )
        else:
            raise TypeError((
                "Input arg 'tokenizer_name_or_path' is not " +
                "BertTokenize model or path to model."
            ))


class Embedder:
    """Class convert sequence from text to dence view.

    @author: Odegov Ilya.

    """

    def __init__(
            self, emberdder, tokenizer, device,
            tokenizer_max_length=512, clean=True):
        """Init class object.

        Args:
            emberdder (BertModel): model for create embeddings.

            tokenizer (BertTokenizer): model for convert text in tokens.s

            device (torch.device): worked device.

            tokenizer_max_length (int, optional): max quantity of tokens.
            Defaults to 512.

            clean (bool, optional): flag for clean text from tag, utl and etc.
            Defaults to True.

        """
        self._device = device
        self.emberdder = emberdder
        self.emberdder.eval()
        self._preprocessor = PreprocessText(
            tokenizer, max_length=tokenizer_max_length, clean=clean
        )

    def _encode(self, tokens):
        """Encode separate tokens sequece

        Args:
            tokens (list, Tensor): sequence of tokens.

        Returns:
            [ndarrya]: text in dence number format.
        """
        result = self._emberdder(
            input_ids=tokens.to(self._device))
        return result[0][0, 0, :].cpu().detach().numpy().reshape(-1)

    def encode(self, text, return_tensors="pt"):
        """Clean, tokenize and encode to dence format.

        Args:
            text (str, Iterable, Iterator, DataFrame): text sequences.

            return_tensors (str, optional): flag for output format.Can be None, pt, tf.
            Defaults to "pt".

        Returns:
            [ndarrya]: text in dence number format for sequences.
        """
        if isinstance(text, str):
            text = [text]
        text = self._preprocessor.preprocess(text, return_tensors)
        text = [self._encode(element) for element in tqdm(text)]
        return asarray(text)

    @property
    def emberdder(self):
        """Getter for embedder model."""
        return self._emberdder

    @emberdder.setter
    def emberdder(self, emberdder):
        """Setter for private class variable."""
        if isinstance(emberdder, BertModel):
            # get already download Bert
            self._emberdder = emberdder.to(self._device)
        elif isinstance(emberdder, str):
            # load from folder or huggingface repo.
            self._emberdder = BertModel.from_pretrained(emberdder).to(self._device)
        elif isinstance(emberdder, Iterable):
            # get path in separate format [folder, model_name]
            self._emberdder = BertModel.from_pretrained(os.path.join(*emberdder))
            self._emberdder = self._emberdder.to(self._device)
        else:
            raise ValueError(
                "Input variable is not model, path to model or path in separate format"
            )


class DataLoader:
    """
    Load train/valid/test data for bert finetune model.

    @author: Odegov Ilya.

    """

    def __init__(self, dataset, shuffle_=True, drop_last=False):
        """Create class object.

        Args:
            dataset (Iterable, Iterator): loading dataset

            shuffle_ (bool, optional): shuffle dataset flag. Defaults to True.

            drop_last (bool, optional): drop last batch flag. Defaults to False.
        """
        self._dataset = dataset
        if drop_last:
            self._dataset = self._dataset[:-1]
        self._shuffle = shuffle_
        self._generator = self._get_next_batch()

    def _get_next_batch(self):
        """Yield batch."""
        dataset = self._dataset
        if self._shuffle:
            shuffle(dataset)
        for batch in dataset:
            yield batch

    def __len__(self):
        """Return dataset len."""
        return len(self._dataset)

    def __iter__(self):
        """Create generator and return it."""
        return self._get_next_batch()

    @property
    def dataset(self):
        """Getter for dataset."""
        return self._dataset

    @property
    def shuffle_(self):
        """Getter for shuffle_."""
        return self._shuffle

if __name__ == "__main__":
    print(convert_from_transilt("zashchishchayushchiesya"))