"""Jalankan seluruh pipeline NLP Fase 5 dan hasilkan artifact + figure.

Output:
- data/processed/nlp/tags_clean.parquet
- data/processed/nlp/movie_tag_docs.parquet
- data/processed/nlp/movie_text_embeddings.npy, movie_text_ids.npy, movie_text.faiss
- data/processed/nlp/bertopic_topics.parquet
- data/processed/nlp/genome_clusters.parquet
- reports/figures/24_..30_*.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from src import nlp as nlpmod

FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
NLP = ROOT / "data" / "processed" / "nlp"
NLP.mkdir(parents=True, exist_ok=True)


def step_clean():
    print("[1/6] clean_tags...")
    df = nlpmod.clean_tags(lemmatize=True)
    print(f"      tags_clean rows={df.height}")
    return df


def step_docs(df):
    print("[2/6] build_movie_tag_docs...")
    docs = nlpmod.build_movie_tag_docs(df, min_count=3)
    print(f"      movies with docs={docs.height}")
    return docs


def step_embed(docs, sample_cap: int = 40000):
    print(f"[3/6] encode movies (cap={sample_cap})...")
    if docs.height > sample_cap:
        docs_s = docs.sort("total_tags", descending=True).head(sample_cap)
    else:
        docs_s = docs
    emb, ids = nlpmod.encode_movies(docs_s)
    index = nlpmod.build_faiss(emb)
    print(f"      embeddings shape={emb.shape}")
    return docs_s, emb, ids, index


def step_bertopic(docs_s):
    print("[4/6] BERTopic...")
    from bertopic import BERTopic
    from sklearn.feature_extraction.text import CountVectorizer

    texts = docs_s["doc"].to_list()
    vec = CountVectorizer(stop_words="english", min_df=5, ngram_range=(1, 2))
    topic_model = BERTopic(
        vectorizer_model=vec,
        calculate_probabilities=False,
        verbose=False,
        min_topic_size=30,
    )
    topics, _ = topic_model.fit_transform(texts)
    info = topic_model.get_topic_info()
    assign = pl.DataFrame(
        {"movieId": docs_s["movieId"].to_numpy(), "topic": np.asarray(topics, dtype=np.int32)}
    )
    assign.write_parquet(NLP / "bertopic_topics.parquet", compression="zstd")
    info.to_csv(NLP / "bertopic_info.csv", index=False)
    print(f"      topics found={info.shape[0]-1} (excluding -1 noise)")
    return topic_model, info


def step_clusters():
    print("[5/6] UMAP + HDBSCAN on genome embedding...")
    return nlpmod.cluster_genome()


def step_cooc(df):
    print("[6/6] co-occurrence network...")
    return nlpmod.tag_cooccurrence(df, top_n=80, min_edge=40)


def fig_tag_freq(df):
    top = (
        df.group_by("tag_clean").agg(pl.len().alias("n"))
        .sort("n", descending=True).head(30)
    )
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(top["tag_clean"][::-1], top["n"][::-1], color="steelblue")
    ax.set_title("Top 30 tag terbanyak (setelah pembersihan)")
    ax.set_xlabel("jumlah rating-tag")
    fig.tight_layout()
    fig.savefig(FIG / "24_top_tags.png", dpi=120)
    plt.close(fig)


def fig_topic_sizes(info):
    head = info[info["Topic"] >= 0].head(20)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(head["Name"][::-1], head["Count"][::-1], color="teal")
    ax.set_title("BERTopic — 20 topik terbesar")
    ax.set_xlabel("jumlah film")
    fig.tight_layout()
    fig.savefig(FIG / "25_bertopic_sizes.png", dpi=120)
    plt.close(fig)


def fig_umap(clusters):
    df = clusters.to_pandas()
    fig, ax = plt.subplots(figsize=(9, 8))
    noise = df[df["cluster"] == -1]
    clustered = df[df["cluster"] >= 0]
    ax.scatter(noise["umap_x"], noise["umap_y"], s=2, c="lightgrey", alpha=0.4, label="noise")
    sc = ax.scatter(
        clustered["umap_x"], clustered["umap_y"],
        s=3, c=clustered["cluster"], cmap="tab20", alpha=0.7,
    )
    ax.set_title(f"UMAP(genome 1128d) + HDBSCAN — {clustered['cluster'].nunique()} cluster")
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    fig.tight_layout()
    fig.savefig(FIG / "26_umap_genome.png", dpi=120)
    plt.close(fig)


def fig_wordclouds_per_genre(df):
    from wordcloud import WordCloud

    movies = pl.read_parquet(ROOT / "data/processed/movies.parquet").select(["movieId", "genres"])
    sub = df.join(movies, on="movieId", how="inner")
    genres = ["Action", "Comedy", "Drama", "Horror", "Romance", "Sci-Fi"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, g in zip(axes.ravel(), genres):
        s = sub.filter(pl.col("genres").str.contains(g, literal=True))
        freq = (
            s.group_by("tag_clean").agg(pl.len().alias("n"))
            .sort("n", descending=True).head(150)
        )
        if freq.height == 0:
            ax.set_title(f"{g} (kosong)"); ax.axis("off"); continue
        wc = WordCloud(width=600, height=400, background_color="white", colormap="viridis")
        wc.generate_from_frequencies(dict(zip(freq["tag_clean"].to_list(), freq["n"].to_list())))
        ax.imshow(wc, interpolation="bilinear"); ax.set_title(g); ax.axis("off")
    fig.suptitle("WordCloud tag per genre", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG / "27_wordcloud_genre.png", dpi=120)
    plt.close(fig)


def fig_cooc_network(top_tags, edges):
    import networkx as nx

    G = nx.Graph()
    G.add_nodes_from(top_tags)
    for a, b, w in edges:
        G.add_edge(a, b, weight=w)
    if G.number_of_edges() == 0:
        return
    pos = nx.spring_layout(G, seed=42, k=0.6, iterations=60)
    deg = dict(G.degree(weight="weight"))
    node_size = [50 + deg[n] * 0.2 for n in G.nodes()]
    edge_widths = [0.2 + G[u][v]["weight"] / 400 for u, v in G.edges()]
    fig, ax = plt.subplots(figsize=(13, 11))
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.35, edge_color="grey")
    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color="#2a7fbf", alpha=0.85)
    # label only top nodes by degree
    topk = sorted(deg.items(), key=lambda x: -x[1])[:40]
    nx.draw_networkx_labels(G, pos, labels={n: n for n, _ in topk}, font_size=8)
    ax.set_title("Co-occurrence network top-80 tag (edge >=40 film bersama)")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIG / "28_tag_cooccurrence.png", dpi=120)
    plt.close(fig)


def fig_semantic_demo(model_name, docs_s, emb, ids, index):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    movies = pl.read_parquet(ROOT / "data/processed/movies.parquet").select(["movieId", "title", "genres"])
    queries = [
        "mind-bending time travel science fiction",
        "feel good romantic comedy",
        "dark psychological thriller",
        "epic space opera with aliens",
        "based on a true story about war",
    ]
    rows = []
    for q in queries:
        hits = nlpmod.semantic_search(q, model, index, ids, k=5)
        for rank, (mid, score) in enumerate(hits, 1):
            mrow = movies.filter(pl.col("movieId") == mid)
            title = mrow["title"][0] if mrow.height else f"id={mid}"
            rows.append({"query": q, "rank": rank, "score": round(score, 3), "title": title})
    out = pl.DataFrame(rows)
    out.write_csv(NLP / "semantic_search_demo.csv")
    print(out.to_pandas().to_string(index=False))


def main():
    df = step_clean()
    docs = step_docs(df)
    docs_s, emb, ids, index = step_embed(docs, sample_cap=30000)
    topic_model, info = step_bertopic(docs_s)
    clusters = step_clusters()
    top_tags, edges = step_cooc(df)

    print("\n[figures]")
    fig_tag_freq(df)
    fig_topic_sizes(info)
    fig_umap(clusters)
    fig_wordclouds_per_genre(df)
    fig_cooc_network(top_tags, edges)
    fig_semantic_demo("sentence-transformers/all-MiniLM-L6-v2", docs_s, emb, ids, index)

    summary = {
        "tags_clean_rows": df.height,
        "movies_with_docs": docs.height,
        "movies_embedded": int(len(ids)),
        "bertopic_n_topics": int((info["Topic"] >= 0).sum()),
        "genome_clusters": int((clusters["cluster"] >= 0).unique().len()),
        "cooccurrence_edges": len(edges),
    }
    (NLP / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\nDONE", summary)


if __name__ == "__main__":
    main()
