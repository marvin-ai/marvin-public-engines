#!/usr/bin/env python
# coding=utf-8

"""AcquisitorAndCleaner engine action.

Use this module to add the project main code.
"""

from .._compatibility import six
from .._logging import get_logger

from marvin_python_toolbox.engine_base import EngineBaseDataHandler

__all__ = ['AcquisitorAndCleaner']


logger = get_logger('acquisitor_and_cleaner')


class AcquisitorAndCleaner(EngineBaseDataHandler):

    def __init__(self, **kwargs):
        super(AcquisitorAndCleaner, self).__init__(**kwargs)

    def execute(self, params, **kwargs):
        import nltk
        import unicodedata
        import pandas as pd
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize
        from marvin_python_toolbox.common.data import MarvinData

        nltk.download('stopwords')
        stop_words = stopwords.words('portuguese')

        initial_dataset = pd.read_csv(
            MarvinData.download_file("https://s3.amazonaws.com/automl-example/produtos.csv"),
            delimiter=";", encoding='utf-8')


        def remove_nonlatin(string):
            new_chars = []
            for char in string:
                if char == '\n':
                    new_chars.append(' ')
                    continue
                try:
                    if unicodedata.name(char).startswith(('LATIN', 'SPACE')):
                        new_chars.append(char)
                except:
                    continue
            return ''.join(new_chars)


        def pre_processor(text):
            stops = set(stopwords.words("portuguese"))
            text = remove_nonlatin(text)
            words = text.lower().split()
            words = ' '.join([w for w in words if not w in stops])
            return words


        initial_dataset["text"] = initial_dataset["nome"] + " " + initial_dataset["descricao"]
        initial_dataset.drop(["descricao", "nome"], axis=1, inplace=True)
        initial_dataset.dropna(inplace=True)
        initial_dataset['text'] = initial_dataset['text'].apply(pre_processor)

        self.marvin_initial_dataset = initial_dataset

