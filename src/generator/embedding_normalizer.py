"""Embedding-based attribute normalizer for domain-agnostic semantic clustering.

This module provides functionality to cluster semantically similar attribute names
using sentence embeddings, enabling domain-agnostic attribute normalization without
hard-coded synonym dictionaries.
"""

import logging
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger("EmbeddingAttributeNormalizer")


class EmbeddingAttributeNormalizer:
    """Clusters similar attribute names using semantic embeddings.

    This class uses sentence transformers to compute embeddings for attribute names,
    then clusters them using agglomerative clustering based on cosine similarity.
    This approach is domain-agnostic and doesn't require hard-coded synonym mappings.

    Attributes:
        model: SentenceTransformer model for computing embeddings
        similarity_threshold: Minimum cosine similarity for clustering (0.0-1.0)
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.85,
    ) -> None:
        """Initialize the EmbeddingAttributeNormalizer.

        Args:
            model_name: Name of the sentence-transformers model to use
            similarity_threshold: Minimum similarity for grouping attributes (0.0-1.0)
        """
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self._model = None  # Lazy loading

    @property
    def model(self):
        """Lazy load the sentence transformer model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                logger.info(f"Loading sentence transformer model: {self.model_name}")
                self._model = SentenceTransformer(self.model_name)
            except ImportError:
                logger.warning(
                    "sentence-transformers not installed. "
                    "Install with: pip install sentence-transformers"
                )
                raise
        return self._model

    def cluster_attributes(
        self,
        attribute_names: List[str],
    ) -> Dict[str, List[str]]:
        """Cluster similar attribute names using embeddings.

        Args:
            attribute_names: List of attribute names to cluster

        Returns:
            Dictionary mapping canonical name to list of variant names
            {canonical_name: [variant1, variant2, ...]}
        """
        if not attribute_names:
            return {}

        # Remove duplicates while preserving order
        unique_names = list(dict.fromkeys(attribute_names))

        if len(unique_names) == 1:
            canonical = self._to_snake_case(unique_names[0])
            return {canonical: unique_names}

        # Prepare names for embedding (convert to readable form)
        readable_names = [self._to_readable_form(name) for name in unique_names]

        # Get embeddings
        try:
            embeddings = self.model.encode(readable_names, convert_to_numpy=True)
        except Exception as e:
            logger.warning(f"Failed to compute embeddings: {e}")
            # Fallback: each name is its own cluster
            return {self._to_snake_case(name): [name] for name in unique_names}

        # Cluster using agglomerative clustering
        clusters = self._agglomerative_cluster(unique_names, embeddings)

        return clusters

    def _agglomerative_cluster(
        self,
        names: List[str],
        embeddings: "np.ndarray",
    ) -> Dict[str, List[str]]:
        """Perform agglomerative clustering on embeddings.

        Args:
            names: Original attribute names
            embeddings: Embedding vectors for each name

        Returns:
            Dictionary mapping canonical name to variants
        """
        try:
            from sklearn.cluster import AgglomerativeClustering
        except ImportError:
            logger.warning(
                "scikit-learn not installed. "
                "Install with: pip install scikit-learn"
            )
            # Fallback: each name is its own cluster
            return {self._to_snake_case(name): [name] for name in names}

        # Distance threshold = 1 - similarity_threshold (for cosine distance)
        distance_threshold = 1.0 - self.similarity_threshold

        try:
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=distance_threshold,
                metric="cosine",
                linkage="average",
            )
            labels = clustering.fit_predict(embeddings)
        except Exception as e:
            logger.warning(f"Clustering failed: {e}")
            # Fallback: each name is its own cluster
            return {self._to_snake_case(name): [name] for name in names}

        # Group names by cluster label
        cluster_groups: Dict[int, List[str]] = {}
        for name, label in zip(names, labels):
            if label not in cluster_groups:
                cluster_groups[label] = []
            cluster_groups[label].append(name)

        # Select canonical name for each cluster
        result: Dict[str, List[str]] = {}
        for variants in cluster_groups.values():
            canonical = self._select_canonical_name(variants)
            result[canonical] = variants

        logger.debug(
            f"Clustered {len(names)} attributes into {len(result)} canonical groups"
        )

        return result

    def _select_canonical_name(self, variants: List[str]) -> str:
        """Select the best canonical name from a cluster of variants.

        Selection criteria:
        1. Prefer snake_case names
        2. Prefer shorter names
        3. Prefer names without redundant prefixes/suffixes

        Args:
            variants: List of attribute name variants

        Returns:
            The selected canonical name in snake_case
        """
        if not variants:
            return ""

        if len(variants) == 1:
            return self._to_snake_case(variants[0])

        scored: List[Tuple[int, str]] = []

        for name in variants:
            snake = self._to_snake_case(name)
            score = 0

            # Prefer snake_case format
            if "_" in name:
                score += 20

            # Prefer shorter names (but not too short)
            length = len(snake)
            if 3 <= length <= 15:
                score += 10
            score -= length  # Penalize very long names

            # Penalize redundant prefixes/suffixes
            if snake.endswith("_name") or snake.endswith("_value"):
                score -= 5
            if snake.startswith("the_") or snake.startswith("a_"):
                score -= 5

            # Prefer common patterns
            common_patterns = [
                "birth_", "death_", "name", "date", "place",
                "occupation", "nationality", "title", "author"
            ]
            for pattern in common_patterns:
                if pattern in snake:
                    score += 5
                    break

            scored.append((score, snake))

        # Sort by score (descending), then alphabetically for ties
        scored.sort(key=lambda x: (-x[0], x[1]))

        return scored[0][1]

    def _to_snake_case(self, name: str) -> str:
        """Convert an attribute name to snake_case.

        Args:
            name: Raw attribute name

        Returns:
            Name in snake_case format
        """
        # Replace spaces and hyphens with underscores
        result = name.lower().strip()
        result = re.sub(r"[\s\-]+", "_", result)

        # Remove special characters
        result = re.sub(r"[^a-z0-9_]", "", result)

        # Remove consecutive underscores
        result = re.sub(r"_+", "_", result)

        # Remove leading/trailing underscores
        result = result.strip("_")

        return result

    def _to_readable_form(self, name: str) -> str:
        """Convert attribute name to readable form for embedding.

        This helps the model understand the semantic meaning better.

        Args:
            name: Attribute name (possibly in snake_case)

        Returns:
            Readable form with spaces
        """
        # Convert snake_case to spaces
        readable = name.replace("_", " ").replace("-", " ")

        # Clean up
        readable = re.sub(r"\s+", " ", readable).strip().lower()

        return readable

    def normalize_attributes_for_entity(
        self,
        entity_type: str,
        attributes: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Normalize attributes for a single entity type using embeddings.

        Args:
            entity_type: Name of the entity type
            attributes: List of raw attribute dictionaries with 'name' key

        Returns:
            List of normalized attribute dictionaries with:
            - canonical_name: Standard attribute name
            - data_type: Inferred data type
            - variants: List of original names
            - description: Combined description
        """
        if not attributes:
            return []

        # Extract names
        names = [attr.get("name", "") for attr in attributes if attr.get("name")]

        if not names:
            return []

        # Cluster similar names
        clusters = self.cluster_attributes(names)

        # Build normalized attributes
        normalized: List[Dict[str, Any]] = []

        for canonical, variants in clusters.items():
            # Find all attributes that match these variants
            matching_attrs = [
                attr for attr in attributes
                if attr.get("name", "") in variants
            ]

            # Combine information from matching attributes
            data_types: Set[str] = set()
            descriptions: List[str] = []

            for attr in matching_attrs:
                if attr.get("data_type"):
                    data_types.add(attr["data_type"])
                if attr.get("description"):
                    descriptions.append(attr["description"])

            # Select best data type (prefer specific types over TEXT)
            data_type = self._select_best_data_type(data_types, canonical)

            # Combine descriptions
            description = "; ".join(set(descriptions)) if descriptions else ""

            normalized.append({
                "canonical_name": canonical,
                "data_type": data_type,
                "variants": variants,
                "description": description,
            })

        logger.debug(
            f"Normalized {len(names)} attributes for '{entity_type}' "
            f"into {len(normalized)} canonical attributes"
        )

        return normalized

    def _select_best_data_type(
        self,
        data_types: Set[str],
        canonical_name: str,
    ) -> str:
        """Select the best data type from candidates.

        Args:
            data_types: Set of candidate data types
            canonical_name: The canonical attribute name (for inference)

        Returns:
            Selected data type string
        """
        # If only one type, use it
        if len(data_types) == 1:
            return data_types.pop()

        # Priority order (most specific first)
        priority = ["DATE", "INTEGER", "BOOLEAN", "FLOAT", "TEXT"]

        for dtype in priority:
            if dtype in data_types:
                return dtype

        # Infer from name patterns
        name = canonical_name.lower()

        if any(p in name for p in ("date", "birth", "death", "published")):
            return "DATE"
        if any(p in name for p in ("year", "count", "number", "age")):
            return "INTEGER"
        if name.startswith("is_") or name.startswith("has_"):
            return "BOOLEAN"

        return "TEXT"
