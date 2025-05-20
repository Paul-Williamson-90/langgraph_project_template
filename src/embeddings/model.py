from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer(
    "mixedbread-ai/mxbai-embed-large-v1",
    truncate_dim=512,
    cache_folder="./model_cache/",
)


def embed_text(texts: list[str] | str) -> list[list[float]] | list[float]:
    output = embedding_model.encode(texts).tolist()
    return output


async def aembed_texts(texts: list[str] | str) -> list[list[float]] | list[float]:
    output = embedding_model.encode(texts).tolist()
    return output
