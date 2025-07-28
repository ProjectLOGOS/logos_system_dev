# File 16: storage/knowledge_store.py (Completed)
"""Fractal Knowledge Store

Persistent storage system for ontological nodes and relationships.
Provides database integration, spatial indexing, and query capabilities
for THŌNOC knowledge representation.

Dependencies: sqlite3, typing, json
"""

from typing import Dict, List, Tuple, Optional, Union, Any, Set
import sqlite3
import json
import logging
import time
import math
import uuid
import os
import heapq
import random # Added for perturbation
from pathlib import Path

# Import from other modules (adjust paths as needed)
# Assuming relative imports work based on the structure
from ..ontology.ontological_node import OntologicalNode
from ..ontology.trinity_vector import TrinityVector
from ..utils.data_structures import FractalPosition, OntologicalRelation

logger = logging.getLogger(__name__)

class KDNode:
    """Node for k-d tree implementation."""

    def __init__(self, node_id: str, point: List[float]):
        """Initialize k-d tree node.

        Args:
            node_id: Ontological node ID
            point: k-dimensional point
        """
        self.node_id = node_id
        self.point = point
        self.left = None
        self.right = None

class KDTree:
    """k-d tree implementation for efficient spatial indexing."""

    def __init__(self, k: int = 3):
        """Initialize k-d tree.

        Args:
            k: Number of dimensions
        """
        self.k = k
        self.root = None

    def insert(self, node_id: str, point: List[float]) -> None:
        """Insert node into k-d tree.

        Args:
            node_id: Ontological node ID
            point: k-dimensional point
        """
        self.root = self._insert(self.root, node_id, point, 0)

    def _insert(self, node: Optional[KDNode], node_id: str, point: List[float], depth: int) -> KDNode:
        """Internal recursive insertion function."""
        if node is None:
            return KDNode(node_id, point)

        axis = depth % self.k

        if point[axis] < node.point[axis]:
            node.left = self._insert(node.left, node_id, point, depth + 1)
        else:
            node.right = self._insert(node.right, node_id, point, depth + 1)

        return node

    def nearest_neighbor(self, point: List[float]) -> Optional[str]:
        """Find nearest neighbor to query point.

        Args:
            point: Query point

        Returns:
            ID of nearest node or None if tree is empty
        """
        if self.root is None:
            return None

        best = [None, float('inf')]  # [node_id, distance]
        self._nearest_neighbor(self.root, point, 0, best)
        return best[0]

    def _nearest_neighbor(self, node: Optional[KDNode], point: List[float], depth: int, best: List[Any]) -> None:
        """Internal recursive nearest neighbor search."""
        if node is None:
            return

        # Calculate distance to current node
        dist_sq = sum((a - b) ** 2 for a, b in zip(node.point, point))

        # Update best if current is closer
        if dist_sq < best[1]:
            best[0] = node.node_id
            best[1] = dist_sq # Store squared distance

        # Recursively search subtrees
        axis = depth % self.k
        diff = point[axis] - node.point[axis]

        # Search closest subtree first
        first, second = (node.left, node.right) if diff < 0 else (node.right, node.left)

        self._nearest_neighbor(first, point, depth + 1, best)

        # Only search other subtree if it could contain points closer than current best
        # Use squared distance for comparison to avoid sqrt
        if diff ** 2 < best[1]:
            self._nearest_neighbor(second, point, depth + 1, best)

    def k_nearest_neighbors(self, point: List[float], k: int) -> List[Tuple[str, float]]:
        """Find k nearest neighbors to query point.

        Args:
            point: Query point
            k: Number of neighbors to return

        Returns:
            List of (node_id, distance) tuples
        """
        if self.root is None:
            return []

        nearest = []  # Max heap (stores (-distance_sq, node_id))
        self._k_nearest_neighbors(self.root, point, 0, nearest, k)

        # Convert to list of (node_id, distance) tuples
        # Calculate actual distance (sqrt) here
        result = [(nid, math.sqrt(-dist_sq)) for dist_sq, nid in nearest]
        result.sort(key=lambda x: x[1]) # Sort by distance
        return result

    def _k_nearest_neighbors(self, node: Optional[KDNode], point: List[float], depth: int,
                            nearest: List[Tuple[float, str]], k: int) -> None:
        """Internal recursive k-nearest neighbors search using a max heap."""
        if node is None:
            return

        # Calculate squared distance to current node
        dist_sq = sum((a - b) ** 2 for a, b in zip(node.point, point))

        # Use max heap logic: push if heap size < k, pushpop if closer than furthest
        if len(nearest) < k:
            heapq.heappush(nearest, (-dist_sq, node.node_id))
        elif -dist_sq > nearest[0][0]:  # If current dist is smaller than largest in heap
            heapq.heapreplace(nearest, (-dist_sq, node.node_id))

        # Recursively search subtrees
        axis = depth % self.k
        diff = point[axis] - node.point[axis]

        # Search closest subtree first
        first, second = (node.left, node.right) if diff < 0 else (node.right, node.left)

        self._k_nearest_neighbors(first, point, depth + 1, nearest, k)

        # Only search other subtree if it could contain points closer than current k-th furthest
        # Use squared distance for comparison
        if len(nearest) < k or diff ** 2 < -nearest[0][0]:
            self._k_nearest_neighbors(second, point, depth + 1, nearest, k)


class FractalKnowledgeStore:
    """Main database class for ontological knowledge storage."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize knowledge store with configuration.

        Args:
            config: Store configuration
        """
        self.config = config or {}
        self.db_path = self.config.get("db_path", "thonoc_knowledge.db")
        self.cache_size = self.config.get("cache_size", 1000) # TODO: Implement cache eviction
        self.persistence_enabled = self.config.get("persistence_enabled", True)

        # Create database directory if needed
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            try:
                os.makedirs(db_dir)
                logger.info(f"Created directory: {db_dir}")
            except OSError as e:
                logger.error(f"Failed to create directory {db_dir}: {e}")
                self.persistence_enabled = False # Disable persistence if dir creation fails

        # Initialize database
        self.conn = self._initialize_database() if self.persistence_enabled else None

        # In-memory indices
        self.trinity_index = KDTree(k=3)
        self.complex_index = KDTree(k=2)
        self.nodes = {}  # Cache of loaded nodes

        # Load existing data into indices if persistent
        if self.persistence_enabled and self.conn:
            self._load_indices_from_db()

        logger.info(f"Fractal Knowledge Store initialized. Persistence: {self.persistence_enabled}. DB: {self.db_path}")

    def _initialize_database(self) -> Optional[sqlite3.Connection]:
        """Initialize SQLite database.

        Returns:
            Database connection or None if persistence disabled or failed
        """
        if not self.persistence_enabled:
            return None

        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("PRAGMA foreign_keys = ON") # Enable foreign keys

            # Create tables if not exist
            with conn:
                # Nodes table stores the full node metadata as JSON for flexibility
                conn.execute('''
                CREATE TABLE IF NOT EXISTS nodes (
                    id TEXT PRIMARY KEY,
                    metadata TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    trinity_e REAL, -- Indexing columns
                    trinity_g REAL,
                    trinity_t REAL,
                    c_real REAL,    -- Indexing columns
                    c_imag REAL,
                    query_text TEXT -- For FTS if needed later
                )
                ''')

                conn.execute('''
                CREATE TABLE IF NOT EXISTS relations (
                    id TEXT PRIMARY KEY, -- Unique ID for the relation itself
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    relation_type TEXT NOT NULL,
                    weight REAL NOT NULL,
                    metadata TEXT,
                    FOREIGN KEY (source_id) REFERENCES nodes(id) ON DELETE CASCADE,
                    FOREIGN KEY (target_id) REFERENCES nodes(id) ON DELETE CASCADE
                )
                ''')

                # Create indices for faster lookups
                conn.execute('CREATE INDEX IF NOT EXISTS idx_nodes_created ON nodes(created_at)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_nodes_trinity ON nodes(trinity_e, trinity_g, trinity_t)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_nodes_complex ON nodes(c_real, c_imag)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_relations_source ON relations(source_id)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_relations_target ON relations(target_id)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_relations_type ON relations(relation_type)')
                # Consider FTS5 index on query_text if text search becomes important
                # conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS nodes_fts USING fts5(query_text, content=nodes, content_rowid=rowid)")


            logger.info(f"Database initialized successfully at {self.db_path}")
            return conn

        except sqlite3.Error as e:
            logger.error(f"Database initialization error at {self.db_path}: {e}")
            self.persistence_enabled = False # Disable persistence on error
            return None

    def _load_indices_from_db(self) -> None:
        """Load existing node data from DB into in-memory k-d trees."""
        if not self.conn:
            return
        logger.info("Loading spatial indices from database...")
        try:
            cursor = self.conn.execute("SELECT id, trinity_e, trinity_g, trinity_t, c_real, c_imag FROM nodes")
            count = 0
            for row in cursor:
                node_id, e, g, t, cr, ci = row
                if e is not None and g is not None and t is not None:
                    self.trinity_index.insert(node_id, [e, g, t])
                if cr is not None and ci is not None:
                    self.complex_index.insert(node_id, [cr, ci])
                count += 1
            logger.info(f"Loaded {count} nodes into spatial indices.")
        except sqlite3.Error as e:
            logger.error(f"Error loading indices from database: {e}")


    def store_node(self, node: OntologicalNode) -> None:
        """Store ontological node in cache and database.

        Args:
            node: Node to store
        """
        if not isinstance(node, OntologicalNode):
            logger.error(f"Attempted to store invalid object type: {type(node)}")
            return

        # Store in memory cache (overwrite if exists)
        self.nodes[node.node_id] = node

        # Update spatial indices
        trinity_tuple = node.trinity_vector.as_tuple() # Use the node's vector
        complex_point = [node.c.real, node.c.imag]

        # TODO: K-d trees typically don't support updates easily.
        # For simplicity now, we assume insert only or rebuild tree on updates.
        # A more robust solution might use R-trees or other dynamic spatial indices.
        self.trinity_index.insert(node.node_id, list(trinity_tuple))
        self.complex_index.insert(node.node_id, complex_point)

        # Persist to database if enabled
        if self.persistence_enabled and self.conn:
            self._persist_node(node)

    def _persist_node(self, node: OntologicalNode) -> None:
        """Persist node to database.

        Args:
            node: Node to persist
        """
        try:
            node_data = node.to_dict()
            metadata_json = json.dumps(node_data)
            created_at = node_data.get("timestamps", {}).get("created", time.time())
            trinity_tuple = node.trinity_vector.as_tuple()
            query_text = node_data.get("data_payload", {}).get("label", "") # Extract query text

            with self.conn:
                self.conn.execute(
                    '''INSERT OR REPLACE INTO nodes
                       (id, metadata, created_at, trinity_e, trinity_g, trinity_t, c_real, c_imag, query_text)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                    (
                        node.node_id,
                        metadata_json,
                        created_at,
                        trinity_tuple[0], trinity_tuple[1], trinity_tuple[2], # Trinity components
                        node.c.real, node.c.imag, # Complex components
                        query_text # Store query text
                    )
                )

            # Persist relationships (assume relationships are added via add_relation method)
            # If relationships are directly manipulated on the node object before storing,
            # you might need to sync them here too.

        except sqlite3.Error as e:
            logger.error(f"Error persisting node {node.node_id}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error persisting node {node.node_id}: {e}")


    def get_node(self, node_id: str) -> Optional[OntologicalNode]:
        """Retrieve node by ID from cache or database.

        Args:
            node_id: Node identifier

        Returns:
            Node or None if not found
        """
        # Check memory cache first
        if node_id in self.nodes:
            return self.nodes[node_id]

        # Check database if persistence enabled
        if self.persistence_enabled and self.conn:
            node = self._load_node_from_db(node_id)
            if node:
                # Add to cache if loaded from DB
                # TODO: Add cache eviction logic if cache size is exceeded
                self.nodes[node_id] = node
                return node

        logger.debug(f"Node {node_id} not found.")
        return None

    def _load_node_from_db(self, node_id: str) -> Optional[OntologicalNode]:
        """Load node from database metadata column.

        Args:
            node_id: Node identifier

        Returns:
            Reconstructed Node or None if not found or error occurs
        """
        if not self.conn: return None
        try:
            cursor = self.conn.execute(
                'SELECT metadata FROM nodes WHERE id = ?', (node_id,)
            )
            row = cursor.fetchone()

            if not row or not row[0]:
                return None

            metadata_json = row[0]
            node_data = json.loads(metadata_json)

            # Reconstruct node using OntologicalNode.from_dict
            node = OntologicalNode.from_dict(node_data)

            # Ensure the ID matches the requested one (from_dict might regenerate it)
            node.node_id = node_id

            # Load relationships associated with this node from the relations table
            self._load_relationships_for_node(node)

            return node

        except json.JSONDecodeError as e:
             logger.error(f"Error decoding JSON for node {node_id}: {e}. Data: {row[0][:100]}...")
             return None
        except sqlite3.Error as e:
            logger.error(f"Database error loading node {node_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error reconstructing node {node_id}: {e}")
            return None

    def _load_relationships_for_node(self, node: OntologicalNode) -> None:
        """Load relationships for a given node from the database.

        Args:
            node: The node object to populate relationships for.
        """
        if not self.conn: return

        # Clear existing relationships if any (to avoid duplicates if called multiple times)
        node.relationships = []
        try:
            cursor = self.conn.execute(
                'SELECT target_id, relation_type, metadata FROM relations WHERE source_id = ?',
                (node.node_id,)
            )
            for target_id, relation_type, metadata_json in cursor:
                metadata = json.loads(metadata_json) if metadata_json else {}
                node.add_relationship(relation_type, target_id, metadata) # Use method to add

        except sqlite3.Error as e:
            logger.error(f"Database error loading relationships for node {node.node_id}: {e}")
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding relationship metadata for node {node.node_id}: {e}")


    def create_node(self,
                   query: str,
                   trinity_vector: Union[TrinityVector, Tuple[float, float, float]],
                   position: Union[complex, FractalPosition],
                   parent_id: Optional[str] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> OntologicalNode:
        """Create and store new ontological node.

        Args:
            query: Original query string or label for the node
            trinity_vector: Trinity vector
            position: Position in fractal space (complex number)
            parent_id: Optional parent node ID for relationship tracking
            metadata: Optional additional metadata for the node itself

        Returns:
            Created and stored node instance
        """
        # Ensure trinity_vector is a TrinityVector object
        if isinstance(trinity_vector, tuple):
            tv_obj = TrinityVector.from_tuple(trinity_vector)
        elif isinstance(trinity_vector, TrinityVector):
            tv_obj = trinity_vector
        else:
            raise TypeError("trinity_vector must be a TrinityVector or Tuple")

        # Ensure position is a complex number
        if isinstance(position, FractalPosition):
            c_value = position.complex
        elif isinstance(position, complex):
            c_value = position
        else:
             raise TypeError("position must be a complex number or FractalPosition")

        # Create the node instance
        node = OntologicalNode(c_value) # ID is generated here

        # Override calculated properties with provided ones
        node.trinity_vector = tv_obj # Use the provided trinity vector
        node.data_payload["label"] = query
        if metadata:
            node.data_payload["metadata"] = metadata # Store additional metadata

        # Store the node (this handles cache and persistence)
        self.store_node(node)

        # Add parent relationship if provided
        if parent_id:
            # Check if parent exists before creating relation
            if self.get_node(parent_id):
                 self.add_relation(node.node_id, parent_id, "derived_from", 1.0, {"creation_time": time.time()})
            else:
                 logger.warning(f"Parent node {parent_id} not found when creating child {node.node_id}")

        return node


    def add_relation(self,
                    source_id: str,
                    target_id: str,
                    relation_type: str,
                    strength: float = 1.0,
                    metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Add or update a relation between two nodes.

        Args:
            source_id: Source node ID
            target_id: Target node ID
            relation_type: Type of relation (e.g., "entails", "implies", "part_of")
            strength: Relation strength or weight [0-1]
            metadata: Optional relation metadata (dictionary)

        Returns:
            True if successful, False otherwise
        """
        # Verify nodes exist (optional, could rely on FK constraints if DB is strict)
        if not self.get_node(source_id) or not self.get_node(target_id):
            logger.error(f"Cannot add relation: Node {source_id} or {target_id} not found.")
            return False

        # Update relationship in the source node object in cache
        source_node = self.nodes.get(source_id) # Get from cache
        if source_node:
             # Check if relation exists, update or add
             found = False
             for i, (rtype, tid, meta) in enumerate(source_node.relationships):
                  if rtype == relation_type and tid == target_id:
                       source_node.relationships[i] = (relation_type, target_id, {**(metadata or {}), "strength": strength})
                       found = True
                       break
             if not found:
                  source_node.add_relationship(relation_type, target_id, {**(metadata or {}), "strength": strength})


        # Persist relation to database if enabled
        if self.persistence_enabled and self.conn:
             relation_id = f"{source_id}_{relation_type}_{target_id}_{uuid.uuid4().hex[:4]}" # Create a unique ID for the relation row
             metadata_json = json.dumps(metadata or {})
             try:
                  with self.conn:
                       self.conn.execute(
                            '''INSERT OR REPLACE INTO relations
                               (id, source_id, target_id, relation_type, weight, metadata)
                               VALUES (?, ?, ?, ?, ?, ?)''',
                            (relation_id, source_id, target_id, relation_type, strength, metadata_json)
                       )
                  return True
             except sqlite3.Error as e:
                  logger.error(f"Error persisting relation {source_id}->{target_id} ({relation_type}): {e}")
                  return False
        elif source_node: # If only in-memory cache is used
             return True
        else: # Should not happen if node check passed, but safety first
             return False


    def get_relations(self,
                     node_id: str,
                     relation_type: Optional[str] = None,
                     direction: str = "outgoing") -> List[OntologicalRelation]:
        """Get node relations from cache or database.

        Args:
            node_id: Node ID
            relation_type: Optional relation type filter
            direction: "outgoing", "incoming" or "both"

        Returns:
            List of OntologicalRelation objects
        """
        relations = []
        node = self.get_node(node_id) # Ensure node is loaded into cache if needed

        # Get from cache (covers outgoing if node is loaded)
        if node and direction in ["outgoing", "both"]:
            for rel_type, target_id, metadata in node.relationships:
                if relation_type is None or rel_type == relation_type:
                    strength = metadata.get("strength", 1.0)
                    relations.append(OntologicalRelation(
                        source_id=node_id,
                        target_id=target_id,
                        relation_type=rel_type,
                        strength=strength,
                        metadata=metadata
                    ))

        # Query database for potentially uncached or incoming relations
        if self.persistence_enabled and self.conn:
            sql_parts = []
            params = []

            # Build query based on direction
            if direction in ["outgoing", "both"]:
                sql_parts.append("source_id = ?")
                params.append(node_id)
            if direction in ["incoming", "both"]:
                sql_parts.append("target_id = ?")
                params.append(node_id)

            if not sql_parts: return [] # Invalid direction

            base_query = f"SELECT source_id, target_id, relation_type, weight, metadata FROM relations WHERE ({' OR '.join(sql_parts)})"
            if relation_type:
                base_query += " AND relation_type = ?"
                params.append(relation_type)

            try:
                cursor = self.conn.execute(base_query, params)
                # Use a set to avoid duplicates if direction is 'both' and relation was in cache
                seen_relations = {(r.source_id, r.target_id, r.relation_type) for r in relations}

                for src, tgt, rtype, weight, meta_json in cursor:
                    if (src, tgt, rtype) not in seen_relations:
                        meta = json.loads(meta_json) if meta_json else {}
                        relations.append(OntologicalRelation(
                            source_id=src,
                            target_id=tgt,
                            relation_type=rtype,
                            strength=weight,
                            metadata=meta
                        ))
                        seen_relations.add((src, tgt, rtype)) # Add DB result to seen set
            except sqlite3.Error as e:
                logger.error(f"Database error retrieving relations for {node_id}: {e}")
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding relation metadata for node {node_id}: {e}")

        return relations


    def find_nearest_by_trinity(self,
                               vector: Union[TrinityVector, Tuple[float, float, float]],
                               k: int = 5) -> List[Tuple[str, float]]:
        """Find nearest nodes by trinity vector using k-d tree.

        Args:
            vector: Trinity vector (object or tuple)
            k: Number of nearest neighbors to return

        Returns:
            List of (node_id, distance) tuples, sorted by distance
        """
        if isinstance(vector, TrinityVector):
            point = list(vector.to_tuple())
        elif isinstance(vector, tuple) and len(vector) == 3:
            point = list(vector)
        else:
             raise TypeError("vector must be a TrinityVector or a 3-tuple")

        # Ensure k is positive
        k = max(1, k)

        return self.trinity_index.k_nearest_neighbors(point, k)


    def find_nearest_by_position(self,
                               position: Union[complex, FractalPosition],
                               k: int = 5) -> List[Tuple[str, float]]:
        """Find nearest nodes by fractal position using k-d tree.

        Args:
            position: Fractal position (complex number or FractalPosition object)
            k: Number of nearest neighbors to return

        Returns:
            List of (node_id, distance) tuples, sorted by distance
        """
        if isinstance(position, FractalPosition):
            point = [position.c_real, position.c_imag]
        elif isinstance(position, complex):
            point = [position.real, position.imag]
        else:
             raise TypeError("position must be complex or FractalPosition")

        # Ensure k is positive
        k = max(1, k)

        return self.complex_index.k_nearest_neighbors(point, k)


    def find_by_query(self, query_term: str, limit: int = 10) -> List[str]:
        """Find nodes by query text using simple LIKE search.

        Args:
            query_term: Search text
            limit: Maximum number of results

        Returns:
            List of matching node IDs
        """
        # Search in-memory cache first (simple label match)
        cached_results = []
        for node_id, node in self.nodes.items():
            label = node.data_payload.get("label", "")
            if query_term.lower() in label.lower():
                 cached_results.append(node_id)
            if len(cached_results) >= limit:
                 return cached_results

        # If not enough results from cache and persistence is enabled, search DB
        db_results = []
        remaining_limit = limit - len(cached_results)
        if self.persistence_enabled and self.conn and remaining_limit > 0:
            try:
                # Using the indexed query_text column
                cursor = self.conn.execute(
                    'SELECT id FROM nodes WHERE query_text LIKE ? LIMIT ?',
                    (f'%{query_term}%', remaining_limit)
                )
                db_results = [row[0] for row in cursor.fetchall() if row[0] not in cached_results]

            except sqlite3.Error as e:
                logger.error(f"Error searching for query term '{query_term}': {e}")

        # Combine results
        return cached_results + db_results

    def create_entailment(self, source_id: str, target_id: str, strength: float) -> bool:
        """Create logical entailment relation between nodes.

        Args:
            source_id: Source node ID (premise)
            target_id: Target node ID (conclusion)
            strength: Entailment strength [0-1]

        Returns:
            True if successful, False otherwise
        """
        if not (0.0 <= strength <= 1.0):
             logger.warning(f"Entailment strength {strength} out of bounds [0,1]. Clamping.")
             strength = max(0.0, min(1.0, strength))

        return self.add_relation(
            source_id=source_id,
            target_id=target_id,
            relation_type="entails",
            strength=strength,
            metadata={"created_at": time.time()}
        )

    def apply_banach_tarski(self,
                           node_id: str,
                           pieces: int = 2,
                           perturbation_scale: float = 0.01,
                           relation_type: str = "decomposition") -> List[str]:
        """
        Apply a conceptual Banach-Tarski-like decomposition.

        This is a metaphor: It 'decomposes' a concept (node) into multiple
        slightly perturbed versions, representing different facets or
        potential interpretations derived from the original.

        Args:
            node_id: ID of the node to decompose.
            pieces: Number of 'pieces' (new nodes) to create.
            perturbation_scale: Scale of random noise added to trinity/position.
            relation_type: Type of relation linking original to pieces.

        Returns:
            List of IDs of the newly created piece nodes.
        """
        source_node = self.get_node(node_id)
        if not source_node:
            logger.error(f"Banach-Tarski application failed: Node {node_id} not found.")
            return []

        new_node_ids = []
        source_label = source_node.data_payload.get("label", node_id)
        source_trinity = source_node.trinity_vector # Get as object
        source_pos = source_node.c

        for i in range(pieces):
            # Perturb trinity vector
            new_e = source_trinity.existence + random.uniform(-perturbation_scale, perturbation_scale)
            new_g = source_trinity.goodness + random.uniform(-perturbation_scale, perturbation_scale)
            new_t = source_trinity.truth + random.uniform(-perturbation_scale, perturbation_scale)
            perturbed_trinity = TrinityVector(new_e, new_g, new_t) # Normalizes automatically

            # Perturb complex position
            perturbed_pos = source_pos + complex(
                random.uniform(-perturbation_scale, perturbation_scale),
                random.uniform(-perturbation_scale, perturbation_scale)
            )

            # Create new node
            piece_label = f"{relation_type.capitalize()} Piece {i+1} of '{source_label}'"
            # Inherit metadata, potentially modifying it
            new_metadata = source_node.data_payload.get("metadata", {}).copy()
            new_metadata["original_node"] = node_id
            new_metadata["piece_index"] = i+1

            try:
                piece_node = self.create_node(
                    query=piece_label,
                    trinity_vector=perturbed_trinity,
                    position=perturbed_pos,
                    parent_id=node_id, # Link back immediately if desired
                    metadata=new_metadata
                )
                # Add specific relation type from source to piece
                self.add_relation(
                    source_id=node_id,
                    target_id=piece_node.node_id,
                    relation_type=relation_type,
                    strength=1.0 / pieces, # Example strength distribution
                    metadata={"piece_index": i+1}
                )
                new_node_ids.append(piece_node.node_id)
                logger.debug(f"Created Banach-Tarski piece {piece_node.node_id} for node {node_id}")

            except Exception as e:
                logger.error(f"Failed to create piece {i+1} for node {node_id}: {e}")


        logger.info(f"Applied conceptual Banach-Tarski to node {node_id}, creating {len(new_node_ids)} pieces.")
        return new_node_ids

    def close(self) -> None:
        """Close the database connection."""
        if self.conn:
            try:
                self.conn.close()
                logger.info("Database connection closed.")
            except sqlite3.Error as e:
                logger.error(f"Error closing database connection: {e}")
        self.conn = None
        self.persistence_enabled = False

    def __del__(self):
        """Ensure connection is closed when object is destroyed."""
        self.close()