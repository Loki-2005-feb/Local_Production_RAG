from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct
)

from data_loader import EMBED_DIM


class QdrantStorage:

    def __init__(
        self,
        url="http://localhost:6333",
        collection="docs",
        dim=EMBED_DIM
    ):

        # -------------------------------------------
        # Create Qdrant Client
        # -------------------------------------------

        self.client = QdrantClient(
            url=url,
            timeout=30
        )

        self.collection = collection

        # -------------------------------------------
        # Create Collection If Missing
        # -------------------------------------------

        if not self.client.collection_exists(
            self.collection
        ):

            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=dim,
                    distance=Distance.COSINE
                ),
            )

    # =================================================
    # UPSERT VECTORS
    # =================================================

    def upsert(
        self,
        ids,
        vectors,
        payloads
    ):

        points = []

        for i in range(len(ids)):

            points.append(
                PointStruct(
                    id=ids[i],
                    vector=vectors[i],
                    payload=payloads[i]
                )
            )

        self.client.upsert(
            collection_name=self.collection,
            points=points
        )

    # =================================================
    # SEARCH
    # =================================================

    def search(
    self,
    query_vector,
    top_k: int = 5
):

        results = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            with_payload=True,
            limit=top_k
        ).points

        contexts = []

        sources = set()

        for r in results:

            payload = getattr(
                r,
                "payload",
                {}
            ) or {}

            text = payload.get(
                "text",
                ""
            )

            source = payload.get(
                "source",
                ""
            )

            if text:

                contexts.append(text)

                sources.add(source)

        return {
            "contexts": contexts,
            "sources": list(sources)
        }