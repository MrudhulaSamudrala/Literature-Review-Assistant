"""Neo4j property graph store that does not require APOC."""

import re
from typing import Any

from llama_index.core.graph_stores.types import ChunkNode, EntityNode, LabelledNode, Relation
from llama_index.core.vector_stores.types import VectorStoreQuery
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.graph_stores.neo4j.neo4j_property_graph import remove_empty_values

_RELATION_TYPE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


def relation_type(label: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]", "_", label.strip())
    if not cleaned or cleaned[0].isdigit():
        cleaned = "RELATED_TO"
    cleaned = cleaned[:64]
    if not _RELATION_TYPE.match(cleaned):
        return "RELATED_TO"
    return cleaned


def primitive_properties(properties: dict[str, Any]) -> dict[str, Any]:
    clean: dict[str, Any] = {}
    for key, value in properties.items():
        if isinstance(value, (str, int, float, bool)):
            clean[key] = value
    return clean


class ApoCFreeNeo4jPropertyGraphStore(Neo4jPropertyGraphStore):
    """PropertyGraphIndex store using MERGE instead of APOC."""

    def __init__(self, username: str, password: str, url: str, database: str = "neo4j") -> None:
        super().__init__(
            username=username,
            password=password,
            url=url,
            database=database,
            refresh_schema=False,
            create_indexes=False,
        )
        self._supports_vector_index = False

    def get_schema(self, refresh: bool = False) -> Any:
        return self.structured_schema or {}

    def upsert_nodes(self, nodes: list[LabelledNode]) -> None:
        for node in nodes:
            properties = primitive_properties(node.properties)
            if isinstance(node, EntityNode):
                self.structured_query(
                    """
                    MERGE (e:`__Node__` {id: $id})
                    SET e:`__Entity__`:Entity, e.name = $name
                    SET e += $properties
                    """,
                    param_map={"id": node.id, "name": node.name, "properties": properties},
                )
                if node.embedding:
                    self.structured_query(
                        """
                        MATCH (e:`__Node__` {id: $id})
                        CALL db.create.setNodeVectorProperty(e, 'embedding', $embedding)
                        RETURN e.id AS id
                        """,
                        param_map={"id": node.id, "embedding": [float(value) for value in node.embedding]},
                    )
                source_id = properties.get("triplet_source_id")
                if source_id:
                    self.structured_query(
                        """
                        MERGE (c:`__Node__` {id: $source_id})
                        MERGE (e:`__Node__` {id: $entity_id})
                        MERGE (c)-[:MENTIONS]->(e)
                        """,
                        param_map={"source_id": source_id, "entity_id": node.id},
                    )
            elif isinstance(node, ChunkNode):
                self.structured_query(
                    """
                    MERGE (c:`__Node__` {id: $id})
                    SET c:Chunk, c.text = $text
                    SET c += $properties
                    """,
                    param_map={"id": node.id, "text": node.text, "properties": properties},
                )

    def upsert_relations(self, relations: list[Relation]) -> None:
        for relation in relations:
            rel_type = relation_type(relation.label)
            self.structured_query(
                f"""
                MERGE (source:`__Node__` {{id: $source_id}})
                MERGE (target:`__Node__` {{id: $target_id}})
                MERGE (source)-[r:`{rel_type}`]->(target)
                SET r += $properties
                """,
                param_map={
                    "source_id": relation.source_id,
                    "target_id": relation.target_id,
                    "properties": primitive_properties(relation.properties),
                },
            )

    def vector_query(self, query: VectorStoreQuery, **kwargs):
        """Search stored embeddings without requiring a Neo4j vector index."""

        if not query.query_embedding:
            return [], []
        data = self.structured_query(
            """
            MATCH (e:`__Entity__`)
            WHERE e.embedding IS NOT NULL AND size(e.embedding) = $dimension
            WITH e, vector.similarity.cosine(e.embedding, $embedding) AS score
            ORDER BY score DESC
            LIMIT toInteger($limit)
            RETURN e.id AS name, e.name AS label, e{.*, embedding: null, id: null} AS properties, score
            """,
            param_map={
                "embedding": query.query_embedding,
                "dimension": len(query.query_embedding),
                "limit": query.similarity_top_k,
            },
        )
        nodes = []
        scores = []
        for record in data or []:
            nodes.append(
                EntityNode(
                    name=record["name"],
                    label=record.get("label") or "Entity",
                    properties=remove_empty_values(record.get("properties") or {}),
                )
            )
            scores.append(record["score"])
        return nodes, scores

    def get_rel_map(self, graph_nodes, depth: int = 2, limit: int = 30, ignore_rels=None):
        ids = [node.id for node in graph_nodes]
        if not ids:
            return []
        depth = max(1, min(int(depth), 4))
        response = self.structured_query(
            f"""
            WITH $ids AS id_list
            UNWIND range(0, size(id_list) - 1) AS idx
            MATCH (e:`__Entity__`)
            WHERE e.id = id_list[idx]
            MATCH p=(e)-[r*1..{depth}]-(other)
            WHERE ALL(rel IN relationships(p) WHERE type(rel) <> 'MENTIONS')
            UNWIND relationships(p) AS rel
            WITH DISTINCT rel, idx
            WITH startNode(rel) AS source, type(rel) AS type, rel{{.*}} AS rel_properties,
                 endNode(rel) AS endNode, idx
            RETURN source.id AS source_id,
                   source{{.*, embedding: null, id: null}} AS source_properties,
                   type,
                   rel_properties,
                   endNode.id AS target_id,
                   endNode{{.*, embedding: null, id: null}} AS target_properties
            ORDER BY idx
            LIMIT toInteger($limit)
            """,
            param_map={"ids": ids, "limit": limit},
        )
        triples = []
        ignored = set(ignore_rels or [])
        for record in response or []:
            if record["type"] in ignored:
                continue
            triples.append(
                [
                    EntityNode(
                        name=record["source_id"],
                        label="Entity",
                        properties=remove_empty_values(record.get("source_properties") or {}),
                    ),
                    Relation(
                        source_id=record["source_id"],
                        target_id=record["target_id"],
                        label=record["type"],
                        properties=remove_empty_values(record.get("rel_properties") or {}),
                    ),
                    EntityNode(
                        name=record["target_id"],
                        label="Entity",
                        properties=remove_empty_values(record.get("target_properties") or {}),
                    ),
                ]
            )
        return triples

    def delete_paper_relations(self, paper_id: str) -> None:
        self.structured_query(
            """
            MATCH ()-[r]->()
            WHERE r.paper_id = $paper_id AND type(r) <> 'MENTIONS'
            DELETE r
            """,
            param_map={"paper_id": paper_id},
        )
