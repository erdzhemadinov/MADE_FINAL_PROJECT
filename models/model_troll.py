import pickle
from time import time

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers.modeling_bert import BertModel

try:
    from BertCosSim import BertCosineSimilraty
    from BertToxic import BertToxicPredict
    from NerExtr import NerExtractor
    from TextStats import TextStatistics
    from classificator import Classificator
except ImportError:
    from models.BertCosSim import BertCosineSimilraty
    from models.BertToxic import BertToxicPredict
    from models.NerExtr import NerExtractor
    from models.TextStats import TextStatistics
    from models.classificator import Classificator


#self._tokenizer_filename = 'tokenizer_new'
#self._model_filename = 'model_new'

class Model:
    """Final troll predict class."""

    def __init__(self, bert=None, bert_tokenizer=None, cpu=False,
                 tokenizer_filename='troll_model', model_filename='troll_model'):
        """Init class object."""
        self._bert = bert
        self._bert_tokenizer = bert_tokenizer
        self._cpu = cpu

        self._tokenizer_filename = tokenizer_filename
        self._model_filename = model_filename

        print("Загрузка моделей\n")
        self._logreg_model = pickle.load(open('logreg_new.pickle', 'rb'))
        self._logreg_inf = pickle.load(open('logreg_low_cls.pickle', 'rb'))
        self._scaler_inf = pickle.load(open('scaler.pickle', 'rb'))
        self._bert_model = bert
        self._bert_tokenizer = bert_tokenizer
        self._classificator = Classificator(self, classificator=self._logreg_inf,
                                            scaler=self._scaler_inf)

        methods = [
            self._load_BERT, self._load_textstats,
            self._load_Ner, self._load_BERT_cossim
        ]
        for run in tqdm(methods):
            run()

    def _load_BERT(self,):
        if not self._bert_model and not self._bert_tokenizer:
            # load bert from file or url
            self._bert_model = AutoModel.from_pretrained(self._model_filename)
            # load bert tokenizer from file or url
            self._bert_tokenizer = AutoTokenizer.from_pretrained(
                self._tokenizer_filename,
                config=AutoConfig.from_pretrained(self._model_filename)
            )

        self._bert_toxic = BertToxicPredict(
            bert_model=self._bert_model, bert_tokenizer=self._bert_tokenizer,
            classificator=self._logreg_model, cpu=self._cpu,
                                  )

    def _load_textstats(self,):

        with open('stop_words.txt', 'r') as f:
            stop_words = f.read().splitlines()

        self._textstats = TextStatistics(stop_words)

    def _load_Ner(self,):
        self._ner_extraction = NerExtractor()

    def _load_BERT_cossim(self,):
        self._bert_cos_sim = BertCosineSimilraty(
            bert_model=self._bert_model, bert_tokenizer=self._bert_tokenizer,
            cpu=self._cpu,
            )

    def create_features(self, question: str, answer: str) -> np.array:
        """Get question, answer and create features for head model."""
        question_features = np.hstack(([
            self._textstats.get_features(question), self._bert_toxic.predict(question),
            self._ner_extraction.get_features(question),
        ]))

        answer_features = np.hstack(([
            self._textstats.get_features(answer), self._bert_toxic.predict(answer),
            self._ner_extraction.get_features(answer),
        ]))

        cosine_similraty = self._bert_cos_sim.predict(question, answer)

        total_features = np.concatenate(
            (question_features, answer_features, cosine_similraty), axis=0
        )
        return total_features.reshape(1, -1)

    def predict(self, question, answer):
        """Compute finall troll prob."""
        t1 = time()
        features = self.create_features(question, answer)
        prob = self._classificator.predict_proba(features)
        print(f"Compute time = {time() - t1}")
        return str(prob[1])

    def _fit_bertolet(self, ):
        self._bert_toxic.fit()
        self._classificator.fit()

    def fit(self, path_to_data):
        self._classificator.fit(data=path_to_data)

    @property
    def bert(self):
        """Getter for embedder model."""
        return self._bert

    @bert.setter
    def bert(self, new_bert):
        """Setter for embedder model."""
        if isinstance(new_bert, BertModel):
            self._bert = new_bert
        else:
            type_ = type(new_bert)
            raise TypeError(
                f"Recive model wrong type. Need {BertModel}, get {type_}"
            )
