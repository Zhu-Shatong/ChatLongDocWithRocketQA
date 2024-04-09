# coding: utf-8
import re
import requests
import json
from pathlib import Path
from copy import deepcopy
import numpy as np
import pandas as pd
from typing import List, Dict, Union, Optional, Any, Literal
from haystack.nodes.retriever.dense import DenseRetriever
from haystack.errors import HaystackError
from haystack.schema import Document, FilterType
from haystack.document_stores import BaseDocumentStore


class RocketQAEmbeddingRetriever(DenseRetriever):
    def __init__(
            self,
            model,
            document_store: Optional[BaseDocumentStore] = None,
            use_gpu: bool = True,
            is_faq: bool = False,
            top_k: int = 10,
            scale_score: bool = True,
            embed_meta_fields: Optional[List[str]] = None,
    ):
        """
        :param model: RocketQA Dual Ecoder model, either zh_dureader_de or zh_dureader_de_v2.
        :param document_store: An instance of DocumentStore from which to retrieve documents.
        :param use_gpu: Whether to use all available GPUs or the CPU. Falls back on CPU if no GPU is available.
        :param is_faq: Whether to retrieve faq question.
        :param top_k: How many documents to return per query.
        :param scale_score: Whether to scale the similarity score to the unit interval (range of [0,1]).
                            If true (default) similarity scores (e.g. cosine or dot_product) which naturally have a different value range will be scaled to a range of [0,1], where 1 means extremely relevant.
                            Otherwise raw similarity scores (e.g. cosine or dot_product) will be used.
        :param embed_meta_fields: Concatenate the provided meta fields and text passage / table to a text pair that is
                                  then used to create the embedding.
                                  This approach is also used in the TableTextRetriever paper and is likely to improve
                                  performance if your titles contain meaningful information for retrieval
                                  (topic, entities etc.).
                                  If no value is provided, a default empty list will be created.
        """
        if embed_meta_fields is None:
            embed_meta_fields = []
        super().__init__()

        self.document_store = document_store
        self.use_gpu = use_gpu
        self.is_faq = is_faq
        self.top_k = top_k
        self.scale_score = scale_score

        self.embedding_encoder = model
        self.embed_meta_fields = embed_meta_fields

    def retrieve(
            self,
            query: str,
            filters: Optional[FilterType] = None,
            top_k: Optional[int] = None,
            index: Optional[str] = None,
            headers: Optional[Dict[str, str]] = None,
            scale_score: Optional[bool] = None,
            document_store: Optional[BaseDocumentStore] = None,
    ) -> List[Document]:
        """
        Scan through documents in DocumentStore and return a small number documents
        that are most relevant to the query.

        :param query: The query
        :param filters: Optional filters to narrow down the search space to documents whose metadata fulfill certain
                        conditions.
                        Filters are defined as nested dictionaries. The keys of the dictionaries can be a logical
                        operator (`"$and"`, `"$or"`, `"$not"`), a comparison operator (`"$eq"`, `"$in"`, `"$gt"`,
                        `"$gte"`, `"$lt"`, `"$lte"`) or a metadata field name.
                        Logical operator keys take a dictionary of metadata field names and/or logical operators as
                        value. Metadata field names take a dictionary of comparison operators as value. Comparison
                        operator keys take a single value or (in case of `"$in"`) a list of values as value.
                        If no logical operator is provided, `"$and"` is used as default operation. If no comparison
                        operator is provided, `"$eq"` (or `"$in"` if the comparison value is a list) is used as default
                        operation.

                            __Example__:

                            ```python
                            filters = {
                                "$and": {
                                    "type": {"$eq": "article"},
                                    "date": {"$gte": "2015-01-01", "$lt": "2021-01-01"},
                                    "rating": {"$gte": 3},
                                    "$or": {
                                        "genre": {"$in": ["economy", "politics"]},
                                        "publisher": {"$eq": "nytimes"}
                                    }
                                }
                            }
                            # or simpler using default operators
                            filters = {
                                "type": "article",
                                "date": {"$gte": "2015-01-01", "$lt": "2021-01-01"},
                                "rating": {"$gte": 3},
                                "$or": {
                                    "genre": ["economy", "politics"],
                                    "publisher": "nytimes"
                                }
                            }
                            ```

                            To use the same logical operator multiple times on the same level, logical operators take
                            optionally a list of dictionaries as value.

                            __Example__:

                            ```python
                            filters = {
                                "$or": [
                                    {
                                        "$and": {
                                            "Type": "News Paper",
                                            "Date": {
                                                "$lt": "2019-01-01"
                                            }
                                        }
                                    },
                                    {
                                        "$and": {
                                            "Type": "Blog Post",
                                            "Date": {
                                                "$gte": "2019-01-01"
                                            }
                                        }
                                    }
                                ]
                            }
                            ```
        :param top_k: How many documents to return per query.
        :param index: The name of the index in the DocumentStore from which to retrieve documents
        :param scale_score: Whether to scale the similarity score to the unit interval (range of [0,1]).
                                           If true similarity scores (e.g. cosine or dot_product) which naturally have a different value range will be scaled to a range of [0,1], where 1 means extremely relevant.
                                           Otherwise raw similarity scores (e.g. cosine or dot_product) will be used.
        :param document_store: the docstore to use for retrieval. If `None`, the one given in the `__init__` is used instead.
        """
        document_store = document_store or self.document_store
        if document_store is None:
            raise ValueError(
                "This Retriever was not initialized with a Document Store. Provide one to the retrieve() method."
            )
        if top_k is None:
            top_k = self.top_k
        if index is None:
            index = document_store.index
        if scale_score is None:
            scale_score = self.scale_score
        query_emb = self.embed_queries(queries=[query])
        documents = document_store.query_by_embedding(
            query_emb=query_emb[0], filters=filters, top_k=top_k, index=index, headers=headers, scale_score=scale_score
        )
        return documents

    def retrieve_batch(
            self,
            queries: List[str],
            filters: Optional[Union[FilterType,
                                    List[Optional[FilterType]]]] = None,
            top_k: Optional[int] = None,
            index: Optional[str] = None,
            headers: Optional[Dict[str, str]] = None,
            batch_size: Optional[int] = None,
            scale_score: Optional[bool] = None,
            document_store: Optional[BaseDocumentStore] = None,
    ) -> List[List[Document]]:
        """
        Scan through documents in DocumentStore and return a small number documents
        that are most relevant to the supplied queries.

        Returns a list of lists of Documents (one per query).

        :param queries: List of query strings.
        :param filters: Optional filters to narrow down the search space to documents whose metadata fulfill certain
                        conditions. Can be a single filter that will be applied to each query or a list of filters
                        (one filter per query).

                        Filters are defined as nested dictionaries. The keys of the dictionaries can be a logical
                        operator (`"$and"`, `"$or"`, `"$not"`), a comparison operator (`"$eq"`, `"$in"`, `"$gt"`,
                        `"$gte"`, `"$lt"`, `"$lte"`) or a metadata field name.
                        Logical operator keys take a dictionary of metadata field names and/or logical operators as
                        value. Metadata field names take a dictionary of comparison operators as value. Comparison
                        operator keys take a single value or (in case of `"$in"`) a list of values as value.
                        If no logical operator is provided, `"$and"` is used as default operation. If no comparison
                        operator is provided, `"$eq"` (or `"$in"` if the comparison value is a list) is used as default
                        operation.

                            __Example__:

                            ```python
                            filters = {
                                "$and": {
                                    "type": {"$eq": "article"},
                                    "date": {"$gte": "2015-01-01", "$lt": "2021-01-01"},
                                    "rating": {"$gte": 3},
                                    "$or": {
                                        "genre": {"$in": ["economy", "politics"]},
                                        "publisher": {"$eq": "nytimes"}
                                    }
                                }
                            }
                            # or simpler using default operators
                            filters = {
                                "type": "article",
                                "date": {"$gte": "2015-01-01", "$lt": "2021-01-01"},
                                "rating": {"$gte": 3},
                                "$or": {
                                    "genre": ["economy", "politics"],
                                    "publisher": "nytimes"
                                }
                            }
                            ```

                            To use the same logical operator multiple times on the same level, logical operators take
                            optionally a list of dictionaries as value.

                            __Example__:

                            ```python
                            filters = {
                                "$or": [
                                    {
                                        "$and": {
                                            "Type": "News Paper",
                                            "Date": {
                                                "$lt": "2019-01-01"
                                            }
                                        }
                                    },
                                    {
                                        "$and": {
                                            "Type": "Blog Post",
                                            "Date": {
                                                "$gte": "2019-01-01"
                                            }
                                        }
                                    }
                                ]
                            }
                            ```
        :param top_k: How many documents to return per query.
        :param index: The name of the index in the DocumentStore from which to retrieve documents
        :param batch_size: Number of queries to embed at a time.
        :param scale_score: Whether to scale the similarity score to the unit interval (range of [0,1]).
                            If true similarity scores (e.g. cosine or dot_product) which naturally have a different
                            value range will be scaled to a range of [0,1], where 1 means extremely relevant.
                            Otherwise raw similarity scores (e.g. cosine or dot_product) will be used.
        :param document_store: the docstore to use for retrieval. If `None`, the one given in the `__init__` is used instead.
        """
        document_store = document_store or self.document_store
        if document_store is None:
            raise ValueError(
                "This Retriever was not initialized with a Document Store. Provide one to the retrieve_batch() method."
            )
        if top_k is None:
            top_k = self.top_k

        if batch_size is None:
            batch_size = 32

        if index is None:
            index = document_store.index
        if scale_score is None:
            scale_score = self.scale_score

        # embed_queries is already batched within by batch_size, so no need to batch the input here
        query_embs: np.ndarray = self.embed_queries(queries=queries)
        batched_query_embs: List[np.ndarray] = []
        for i in range(0, len(query_embs), batch_size):
            batched_query_embs.extend(query_embs[i: i + batch_size])
        documents = document_store.query_by_embedding_batch(
            query_embs=batched_query_embs,
            top_k=top_k,
            filters=filters,
            index=index,
            headers=headers,
            scale_score=scale_score,
        )

        return documents

    def is_need_translate(self, query):
        res = re.findall(r"[a-zA-Z]+", query)
        return len(res) >= 5

    def translate_to_zh(self, query):
        translated_text = ''
        url = "https://n-sino-thirdpart.meetsocial.cn/translate/translateText"
        headers = {
            "Content-Type": "application/json",
            "appId": "alg"
        }
        body = {
            "queryText": query,
            "source": "en",
            "target": "zh-CN"
        }
        res = requests.post(url, headers=headers, json=body)
        if res.status_code == 200:
            json_data = json.loads(res.text)
            if json_data["success"]:
                translated_text = json_data["result"]["text"].strip()

        return translated_text if translated_text else query

    def embed_queries(self, queries: List[str]) -> np.ndarray:
        """
        Create embeddings for a list of queries.

        :param queries: List of queries to embed.
        :return: Embeddings, one per input query, shape: (queries, embedding_dim)
        """
        # for backward compatibility: cast pure str input
        if isinstance(queries, str):
            queries = [queries]
        assert isinstance(
            queries, list), "Expecting a list of texts, i.e. create_embeddings(texts=['text1',...])"
        queries = [self.translate_to_zh(query) if self.is_need_translate(query) else query
                   for query in queries]
        q_embs = self.embedding_encoder.encode_query(query=queries)
        q_embs = np.array([v / np.linalg.norm(v) for v in q_embs])
        return q_embs

    def embed_documents(self, documents: List[Document]) -> np.ndarray:
        """
        Create embeddings for a list of documents.

        :param documents: List of documents to embed.
        :return: Embeddings, one per input document, shape: (docs, embedding_dim)
        """
        documents = self._preprocess_documents(documents)
        if self.is_faq:
            queries = [self.translate_to_zh(doc.content) if self.is_need_translate(doc.content) else doc.content
                       for doc in documents]
            p_embs = self.embedding_encoder.encode_query(query=queries)
            p_embs = np.array([v / np.linalg.norm(v) for v in p_embs])
        else:
            p_embs = self.embedding_encoder.encode_para(
                para=[doc.content for doc in documents])
            p_embs = np.array([v / np.linalg.norm(v) for v in p_embs])
        return p_embs

    def _preprocess_documents(self, docs: List[Document]) -> List[Document]:
        """
        Turns table documents into text documents by representing the table in csv format.
        This allows us to use text embedding models for table retrieval.
        It also concatenates specified meta data fields with the text representations.

        :param docs: List of documents to linearize. If the document is not a table, it is returned as is.
        :return: List of documents with meta data + linearized tables or original documents if they are not tables.
        """
        linearized_docs = []
        for doc in docs:
            doc = deepcopy(doc)
            if doc.content_type == "table":
                if isinstance(doc.content, pd.DataFrame):
                    doc.content = doc.content.to_csv(index=False)
                else:
                    raise HaystackError(
                        "Documents of type 'table' need to have a pd.DataFrame as content field")
            meta_data_fields = [
                doc.meta[key] for key in self.embed_meta_fields if key in doc.meta and doc.meta[key]]
            doc.content = "\n".join(meta_data_fields + [doc.content])
            linearized_docs.append(doc)
        return linearized_docs

    def train(
            self,
            training_data: List[Dict[str, Any]],
            learning_rate: float = 2e-5,
            n_epochs: int = 1,
            num_warmup_steps: Optional[int] = None,
            batch_size: int = 16,
            train_loss: Literal["mnrl", "margin_mse"] = "mnrl",
            num_workers: int = 0,
            use_amp: bool = False,
            **kwargs,
    ) -> None:
        raise NotImplementedError(
            "RocketQAEmbeddingRetriever.train have not implemented")

    def save(self, save_dir: Union[Path, str]) -> None:
        raise NotImplementedError(
            "RocketQAEmbeddingRetriever.save have not implemented")
