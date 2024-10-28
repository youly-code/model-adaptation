import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE, MDS
from sklearn.decomposition import PCA, TruncatedSVD
from umap import UMAP
import seaborn as sns
from sentence_transformers import SentenceTransformer
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Tuple
import pandas as pd


class SemanticSpaceVisualizer:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        return self.model.encode(texts)

    def reduce_dimensions(
        self, embeddings: np.ndarray, method: str = "tsne", n_components: int = 2
    ) -> np.ndarray:
        """Reduce dimensionality using specified method."""
        if method == "tsne":
            # Set perplexity to n_samples/3 (or minimum of 5)
            perplexity = min(max(5, embeddings.shape[0] // 3), 30)
            reducer = TSNE(n_components=n_components, random_state=42, perplexity=perplexity)
        elif method == "pca":
            reducer = PCA(n_components=n_components)
        elif method == "umap":
            reducer = UMAP(n_components=n_components, random_state=42)
        elif method == "mds":
            reducer = MDS(n_components=n_components, random_state=42)
        elif method == "svd":
            reducer = TruncatedSVD(n_components=n_components)
        else:
            raise ValueError(f"Unknown reduction method: {method}")

        return reducer.fit_transform(embeddings)

    def visualize_2d_scatter(
        self,
        texts: List[str],
        categories: List[str] = None,
        method: str = "tsne",
        title: str = None,
    ):
        """Create a 2D scatter plot of semantic space."""
        # Generate embeddings
        embeddings = self.get_embeddings(texts)

        # Reduce to 2D
        coords_2d = self.reduce_dimensions(embeddings, method=method)

        # Create DataFrame
        df = pd.DataFrame(
            {
                "x": coords_2d[:, 0],
                "y": coords_2d[:, 1],
                "text": texts,
                "category": categories or [""] * len(texts),
            }
        )

        # Create plot
        plt.figure(figsize=(12, 8))
        if categories:
            sns.scatterplot(data=df, x="x", y="y", hue="category", style="category")
        else:
            plt.scatter(df["x"], df["y"])

        # Add labels
        for idx, row in df.iterrows():
            plt.annotate(
                row["text"],
                (row["x"], row["y"]),
                xytext=(5, 5),
                textcoords="offset points",
            )

        plt.title(title or f"Semantic Space Visualization using {method.upper()}")
        plt.show()

    def visualize_3d_interactive(
        self, texts: List[str], categories: List[str] = None, method: str = "tsne"
    ):
        """Create an interactive 3D visualization using Plotly."""
        # Generate embeddings
        embeddings = self.get_embeddings(texts)

        # Reduce to 3D
        coords_3d = self.reduce_dimensions(embeddings, method=method, n_components=3)

        # Create DataFrame
        df = pd.DataFrame(
            {
                "x": coords_3d[:, 0],
                "y": coords_3d[:, 1],
                "z": coords_3d[:, 2],
                "text": texts,
                "category": categories or [""] * len(texts),
            }
        )

        # Create 3D scatter plot
        fig = px.scatter_3d(
            df,
            x="x",
            y="y",
            z="z",
            color="category" if categories else None,
            text="text",
            title=f"3D Semantic Space ({method.upper()})",
        )

        fig.update_traces(textposition="top center")
        fig.show()

    def visualize_semantic_heatmap(
        self, texts: List[str], categories: List[str] = None
    ):
        """Create a heatmap of semantic similarities."""
        # Generate embeddings
        embeddings = self.get_embeddings(texts)

        # Calculate similarity matrix
        similarity_matrix = np.inner(embeddings, embeddings)

        # Create heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            similarity_matrix, xticklabels=texts, yticklabels=texts, cmap="viridis"
        )
        plt.title("Semantic Similarity Heatmap")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()

    def visualize_cluster_dendogram(self, texts: List[str]):
        """Create a hierarchical clustering dendogram."""
        # Generate embeddings
        embeddings = self.get_embeddings(texts)

        # Perform hierarchical clustering
        from scipy.cluster.hierarchy import dendrogram, linkage

        linked = linkage(embeddings, "ward")

        plt.figure(figsize=(12, 8))
        dendrogram(linked, labels=texts, leaf_rotation=90)
        plt.title("Hierarchical Clustering of Semantic Space")
        plt.tight_layout()
        plt.show()

    def demonstrate_all_visualizations(self):
        """Run all visualization demos with example data."""
        # Example data
        texts = [
            # Technology cluster
            "artificial intelligence",
            "machine learning",
            "deep neural networks",
            "computer vision",
            # Nature cluster
            "forest ecosystem",
            "wildlife conservation",
            "natural habitat",
            "biodiversity",
            # Space cluster
            "galaxy formation",
            "planetary system",
            "cosmic radiation",
            "stellar evolution",
        ]

        categories = ["Technology"] * 4 + ["Nature"] * 4 + ["Space"] * 4

        # 2D visualizations with different reduction methods
        for method in ["tsne", "pca", "umap", "mds"]:
            self.visualize_2d_scatter(texts, categories, method=method)

        # 3D interactive visualization
        self.visualize_3d_interactive(texts, categories)

        # Heatmap
        self.visualize_semantic_heatmap(texts)

        # Dendogram
        self.visualize_cluster_dendogram(texts)

    def visualize_word_analogies(self, analogies: List[Tuple[str, str, str, str]]):
        """Visualize word analogies in semantic space."""
        # Get all unique words from analogies
        all_words = list({word for analogy in analogies for word in analogy})

        # Get embeddings
        embeddings = self.get_embeddings(all_words)

        # Create a mapping from words to embeddings
        word_to_embedding = dict(zip(all_words, embeddings))

        # Reduce dimensionality for visualization
        coords_2d = self.reduce_dimensions(embeddings, method="pca")
        word_to_coords = dict(zip(all_words, coords_2d))

        # Plot
        plt.figure(figsize=(12, 8))

        # Plot points
        for word, coords in word_to_coords.items():
            plt.scatter(coords[0], coords[1])
            plt.annotate(word, coords, xytext=(5, 5), textcoords="offset points")

        # Plot analogy arrows
        for a1, a2, b1, b2 in analogies:
            # Draw arrow for first pair
            plt.arrow(
                word_to_coords[a1][0],
                word_to_coords[a1][1],
                word_to_coords[a2][0] - word_to_coords[a1][0],
                word_to_coords[a2][1] - word_to_coords[a1][1],
                color="blue",
                alpha=0.3,
            )

            # Draw arrow for second pair
            plt.arrow(
                word_to_coords[b1][0],
                word_to_coords[b1][1],
                word_to_coords[b2][0] - word_to_coords[b1][0],
                word_to_coords[b2][1] - word_to_coords[b1][1],
                color="red",
                alpha=0.3,
            )

        plt.title("Word Analogies in Semantic Space")
        plt.show()

    def demonstrate_analogies(self):
        """Demonstrate word analogies visualization."""
        analogies = [
            ("king", "queen", "man", "woman"),
            ("dog", "puppy", "cat", "kitten"),
            ("good", "better", "bad", "worse"),
        ]
        self.visualize_word_analogies(analogies)


if __name__ == "__main__":
    visualizer = SemanticSpaceVisualizer()

    print("Running all visualization demos...")
    visualizer.demonstrate_all_visualizations()

    print("\nDemonstrating word analogies...")
    visualizer.demonstrate_analogies()
