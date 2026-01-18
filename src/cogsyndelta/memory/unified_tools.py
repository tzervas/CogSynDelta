"""
Unified Tools and Utilities System

Provides native tools for the model system to:
1. Form and manage memories with rich metadata
2. Capture relevance, temporal, and contextual data
3. Extract and store semantic residuals
4. Track skills, tools, and memories with schemas
5. Search and recall with multi-modal queries
6. Optimize memory and computational efficiency

This acts as the "toolbox" that all model sections can access.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Set, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import json
import hashlib
import numpy as np
from collections import defaultdict


class MemoryType(str, Enum):
    """Types of memories that can be stored."""
    EPISODIC = "episodic"  # Specific events/experiences
    SEMANTIC = "semantic"  # General knowledge/facts
    PROCEDURAL = "procedural"  # Skills/how-to knowledge
    WORKING = "working"  # Temporary working memory
    SKILL = "skill"  # Learned skills
    TOOL = "tool"  # Available tools/functions
    CONTEXT = "context"  # Contextual information


class RelevanceLevel(str, Enum):
    """Relevance levels for memories."""
    CRITICAL = "critical"  # 1.0
    HIGH = "high"  # 0.8
    MEDIUM = "medium"  # 0.5
    LOW = "low"  # 0.2
    MINIMAL = "minimal"  # 0.1


@dataclass
class TemporalMetadata:
    """Temporal information for memories."""
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    decay_factor: float = 1.0  # Decreases over time
    retention_priority: float = 1.0
    temporal_window: Optional[Tuple[datetime, datetime]] = None
    
    def update_access(self) -> None:
        """Update access time and count."""
        self.last_accessed = datetime.now()
        self.access_count += 1
    
    def compute_decay(self, current_time: datetime) -> float:
        """Compute temporal decay based on time since creation."""
        time_diff = (current_time - self.created_at).total_seconds()
        # Exponential decay: e^(-λt)
        lambda_decay = 0.0001  # Decay rate
        self.decay_factor = np.exp(-lambda_decay * time_diff)
        return self.decay_factor
    
    def to_dict(self) -> Dict:
        return {
            'created_at': self.created_at.isoformat(),
            'last_accessed': self.last_accessed.isoformat(),
            'access_count': self.access_count,
            'decay_factor': self.decay_factor,
            'retention_priority': self.retention_priority
        }


@dataclass
class ContextualMetadata:
    """Contextual information for memories."""
    source: str  # Where memory came from
    tags: Set[str] = field(default_factory=set)
    related_memories: Set[str] = field(default_factory=set)
    causality: Optional[Dict[str, str]] = None  # cause -> effect
    dependencies: Set[str] = field(default_factory=set)
    location: Optional[str] = None
    user_context: Optional[Dict[str, Any]] = None
    
    def add_tag(self, tag: str) -> None:
        """Add contextual tag."""
        self.tags.add(tag)
    
    def link_memory(self, memory_id: str) -> None:
        """Link to related memory."""
        self.related_memories.add(memory_id)
    
    def to_dict(self) -> Dict:
        return {
            'source': self.source,
            'tags': list(self.tags),
            'related_memories': list(self.related_memories),
            'causality': self.causality,
            'dependencies': list(self.dependencies),
            'location': self.location
        }


@dataclass
class SemanticResidual:
    """Semantic residual capturing fine-grained details."""
    residual_embedding: torch.Tensor
    importance: float
    compression_loss: float
    detail_level: str  # "fine", "medium", "coarse"
    
    def to_dict(self) -> Dict:
        return {
            'shape': list(self.residual_embedding.shape),
            'importance': self.importance,
            'compression_loss': self.compression_loss,
            'detail_level': self.detail_level
        }


@dataclass
class MemorySchema:
    """Complete schema for a memory entry."""
    memory_id: str
    memory_type: MemoryType
    embedding: torch.Tensor
    semantic_residual: Optional[SemanticResidual]
    temporal_metadata: TemporalMetadata
    contextual_metadata: ContextualMetadata
    relevance: float
    confidence: float
    compressed: bool = False
    
    def to_dict(self) -> Dict:
        """Convert to dictionary (for JSON serialization)."""
        return {
            'memory_id': self.memory_id,
            'memory_type': self.memory_type.value,
            'embedding_shape': list(self.embedding.shape),
            'has_residual': self.semantic_residual is not None,
            'temporal': self.temporal_metadata.to_dict(),
            'contextual': self.contextual_metadata.to_dict(),
            'relevance': self.relevance,
            'confidence': self.confidence,
            'compressed': self.compressed
        }


@dataclass
class SkillSchema:
    """Schema for learned skills."""
    skill_id: str
    skill_name: str
    skill_type: str  # "code_generation", "problem_solving", etc.
    proficiency: float  # 0.0-1.0
    usage_count: int
    success_rate: float
    parameters: Dict[str, Any]
    learned_at: datetime
    last_used: datetime
    embedding: Optional[torch.Tensor] = None
    
    def update_usage(self, success: bool) -> None:
        """Update skill usage statistics."""
        self.usage_count += 1
        self.last_used = datetime.now()
        
        # Update success rate (running average)
        alpha = 0.1  # Learning rate
        self.success_rate = (1 - alpha) * self.success_rate + alpha * (1.0 if success else 0.0)
        
        # Update proficiency
        if success:
            self.proficiency = min(1.0, self.proficiency + 0.01)
        else:
            self.proficiency = max(0.0, self.proficiency - 0.005)
    
    def to_dict(self) -> Dict:
        return {
            'skill_id': self.skill_id,
            'skill_name': self.skill_name,
            'skill_type': self.skill_type,
            'proficiency': self.proficiency,
            'usage_count': self.usage_count,
            'success_rate': self.success_rate,
            'parameters': self.parameters,
            'learned_at': self.learned_at.isoformat(),
            'last_used': self.last_used.isoformat()
        }


@dataclass
class ToolSchema:
    """Schema for available tools."""
    tool_id: str
    tool_name: str
    description: str
    function: Callable
    parameters: Dict[str, Any]
    return_type: str
    usage_count: int = 0
    average_latency: float = 0.0
    success_rate: float = 1.0
    
    def record_usage(self, latency: float, success: bool) -> None:
        """Record tool usage."""
        self.usage_count += 1
        
        # Update average latency
        alpha = 0.1
        self.average_latency = (1 - alpha) * self.average_latency + alpha * latency
        
        # Update success rate
        self.success_rate = (1 - alpha) * self.success_rate + alpha * (1.0 if success else 0.0)
    
    def to_dict(self) -> Dict:
        return {
            'tool_id': self.tool_id,
            'tool_name': self.tool_name,
            'description': self.description,
            'parameters': self.parameters,
            'return_type': self.return_type,
            'usage_count': self.usage_count,
            'average_latency_ms': self.average_latency * 1000,
            'success_rate': self.success_rate
        }


class RelevanceScorer(nn.Module):
    """
    Neural network for scoring memory relevance.
    
    Learns to predict how relevant a memory is given:
    - Query context
    - Memory embedding
    - Temporal factors
    - Access patterns
    """
    
    def __init__(self, embed_dim: int = 512) -> None:
        super(RelevanceScorer, self).__init__()
        
        self.embed_dim = embed_dim
        
        # Relevance network
        self.relevance_net = nn.Sequential(
            nn.Linear(embed_dim * 2 + 10, embed_dim),  # +10 for metadata features
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, query: torch.Tensor, memory: torch.Tensor,
               metadata_features: torch.Tensor) -> torch.Tensor:
        """
        Score relevance of memory to query.
        
        Args:
            query: Query embedding [batch, embed_dim]
            memory: Memory embedding [batch, embed_dim]
            metadata_features: Metadata features [batch, 10]
                - decay_factor
                - access_count (normalized)
                - time_since_creation (normalized)
                - time_since_access (normalized)
                - confidence
                - relevance (prior)
                - ...
        
        Returns:
            Relevance scores [batch, 1]
        """
        combined = torch.cat([query, memory, metadata_features], dim=-1)
        relevance_score = self.relevance_net(combined)
        return relevance_score


class SemanticSearchEngine(nn.Module):
    """
    Multi-modal semantic search with learned ranking.
    
    Supports:
    - Embedding-based search
    - Tag-based filtering
    - Temporal filtering
    - Hybrid ranking
    """
    
    def __init__(self, embed_dim: int = 512) -> None:
        super(SemanticSearchEngine, self).__init__()
        
        self.embed_dim = embed_dim
        self.relevance_scorer = RelevanceScorer(embed_dim)
        
        # Query encoder (transforms query to search space)
        self.query_encoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )
    
    def search(self, query: torch.Tensor,
              memories: List[MemorySchema],
              filters: Optional[Dict[str, Any]] = None,
              top_k: int = 10) -> List[Tuple[MemorySchema, float]]:
        """
        Search memories with multi-modal query.
        
        Args:
            query: Query embedding
            memories: List of memory schemas
            filters: Optional filters (tags, time_range, memory_type)
            top_k: Number of results to return
            
        Returns:
            List of (memory, score) tuples sorted by relevance
        """
        # Encode query
        encoded_query = self.query_encoder(query)
        
        # Apply filters
        filtered_memories = self._apply_filters(memories, filters)
        
        if not filtered_memories:
            return []
        
        # Score relevance
        scores = []
        for memory in filtered_memories:
            # Extract metadata features
            metadata_features = self._extract_metadata_features(memory)
            
            # Compute relevance score
            score = self.relevance_scorer(
                encoded_query.unsqueeze(0),
                memory.embedding.unsqueeze(0),
                metadata_features.unsqueeze(0)
            )
            
            # Combine with cosine similarity
            cos_sim = F.cosine_similarity(
                encoded_query.unsqueeze(0),
                memory.embedding.unsqueeze(0),
                dim=-1
            )
            
            # Hybrid score
            final_score = 0.6 * score.item() + 0.4 * cos_sim.item()
            scores.append((memory, final_score))
        
        # Sort by score
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return scores[:top_k]
    
    def _apply_filters(self, memories: List[MemorySchema],
                      filters: Optional[Dict[str, Any]]) -> List[MemorySchema]:
        """Apply filters to memory list."""
        if not filters:
            return memories
        
        filtered = memories
        
        # Memory type filter
        if 'memory_type' in filters:
            filtered = [m for m in filtered if m.memory_type == filters['memory_type']]
        
        # Tag filter
        if 'tags' in filters:
            required_tags = set(filters['tags'])
            filtered = [m for m in filtered 
                       if required_tags.issubset(m.contextual_metadata.tags)]
        
        # Time range filter
        if 'time_range' in filters:
            start, end = filters['time_range']
            filtered = [m for m in filtered 
                       if start <= m.temporal_metadata.created_at <= end]
        
        # Relevance threshold
        if 'min_relevance' in filters:
            filtered = [m for m in filtered 
                       if m.relevance >= filters['min_relevance']]
        
        return filtered
    
    def _extract_metadata_features(self, memory: MemorySchema) -> torch.Tensor:
        """Extract numerical features from metadata."""
        now = datetime.now()
        
        time_since_creation = (now - memory.temporal_metadata.created_at).total_seconds() / 86400  # Days
        time_since_access = (now - memory.temporal_metadata.last_accessed).total_seconds() / 3600  # Hours
        
        features = torch.tensor([
            memory.temporal_metadata.decay_factor,
            min(memory.temporal_metadata.access_count / 100, 1.0),  # Normalized
            min(time_since_creation / 365, 1.0),  # Normalized to years
            min(time_since_access / 168, 1.0),  # Normalized to weeks
            memory.confidence,
            memory.relevance,
            float(memory.compressed),
            float(memory.semantic_residual is not None),
            float(len(memory.contextual_metadata.tags)) / 10,  # Normalized
            float(len(memory.contextual_metadata.related_memories)) / 10  # Normalized
        ])
        
        return features


class UnifiedMemoryManager:
    """
    Unified manager for all types of memories, skills, and tools.
    
    Provides native interface for model to:
    - Store memories with rich metadata
    - Search and recall
    - Manage skills and tools
    - Track usage patterns
    """
    
    def __init__(self, embed_dim: int = 512) -> None:
        self.embed_dim = embed_dim
        
        # Storage
        self.memories: Dict[str, MemorySchema] = {}
        self.skills: Dict[str, SkillSchema] = {}
        self.tools: Dict[str, ToolSchema] = {}
        
        # Search engine
        self.search_engine = SemanticSearchEngine(embed_dim)
        
        # Indices for fast lookup
        self.tag_index: Dict[str, Set[str]] = defaultdict(set)  # tag -> memory_ids
        self.type_index: Dict[MemoryType, Set[str]] = defaultdict(set)  # type -> memory_ids
        
        # Statistics
        self.stats = {
            'total_memories': 0,
            'total_skills': 0,
            'total_tools': 0,
            'total_searches': 0,
            'average_search_time': 0.0
        }
    
    def store_memory(self, embedding: torch.Tensor,
                    memory_type: MemoryType,
                    source: str,
                    relevance: float = 0.5,
                    confidence: float = 1.0,
                    tags: Optional[Set[str]] = None,
                    semantic_residual: Optional[SemanticResidual] = None) -> str:
        """
        Store a memory with complete metadata.
        
        Args:
            embedding: Memory embedding
            memory_type: Type of memory
            source: Source of memory
            relevance: Initial relevance score
            confidence: Confidence in memory
            tags: Optional tags
            semantic_residual: Optional residual for fine details
            
        Returns:
            Memory ID
        """
        # Generate memory ID
        memory_id = hashlib.sha256(
            f"{datetime.now().isoformat()}_{source}".encode()
        ).hexdigest()[:16]
        
        # Create temporal metadata
        now = datetime.now()
        temporal = TemporalMetadata(
            created_at=now,
            last_accessed=now,
            access_count=0,
            retention_priority=relevance
        )
        
        # Create contextual metadata
        contextual = ContextualMetadata(
            source=source,
            tags=tags or set()
        )
        
        # Create memory schema
        memory = MemorySchema(
            memory_id=memory_id,
            memory_type=memory_type,
            embedding=embedding.detach(),
            semantic_residual=semantic_residual,
            temporal_metadata=temporal,
            contextual_metadata=contextual,
            relevance=relevance,
            confidence=confidence
        )
        
        # Store
        self.memories[memory_id] = memory
        
        # Update indices
        self.type_index[memory_type].add(memory_id)
        for tag in contextual.tags:
            self.tag_index[tag].add(memory_id)
        
        # Update stats
        self.stats['total_memories'] += 1
        
        return memory_id
    
    def recall_memory(self, query: torch.Tensor,
                     filters: Optional[Dict[str, Any]] = None,
                     top_k: int = 10) -> List[Tuple[MemorySchema, float]]:
        """
        Recall memories matching query.
        
        Args:
            query: Query embedding
            filters: Optional filters
            top_k: Number of results
            
        Returns:
            List of (memory, score) tuples
        """
        import time
        start = time.time()
        
        # Search
        results = self.search_engine.search(
            query,
            list(self.memories.values()),
            filters=filters,
            top_k=top_k
        )
        
        # Update access metadata
        for memory, score in results:
            memory.temporal_metadata.update_access()
        
        # Update stats
        elapsed = time.time() - start
        self.stats['total_searches'] += 1
        alpha = 0.1
        self.stats['average_search_time'] = (
            (1 - alpha) * self.stats['average_search_time'] + alpha * elapsed
        )
        
        return results
    
    def register_skill(self, skill_name: str, skill_type: str,
                      parameters: Dict[str, Any],
                      embedding: Optional[torch.Tensor] = None) -> str:
        """Register a learned skill."""
        skill_id = f"skill_{hashlib.sha256(skill_name.encode()).hexdigest()[:12]}"
        
        skill = SkillSchema(
            skill_id=skill_id,
            skill_name=skill_name,
            skill_type=skill_type,
            proficiency=0.5,  # Initial proficiency
            usage_count=0,
            success_rate=0.5,
            parameters=parameters,
            learned_at=datetime.now(),
            last_used=datetime.now(),
            embedding=embedding
        )
        
        self.skills[skill_id] = skill
        self.stats['total_skills'] += 1
        
        return skill_id
    
    def register_tool(self, tool_name: str, description: str,
                     function: Callable, parameters: Dict[str, Any],
                     return_type: str) -> str:
        """Register a tool/function."""
        tool_id = f"tool_{hashlib.sha256(tool_name.encode()).hexdigest()[:12]}"
        
        tool = ToolSchema(
            tool_id=tool_id,
            tool_name=tool_name,
            description=description,
            function=function,
            parameters=parameters,
            return_type=return_type
        )
        
        self.tools[tool_id] = tool
        self.stats['total_tools'] += 1
        
        return tool_id
    
    def use_tool(self, tool_id: str, **kwargs) -> Any:
        """Execute a tool and record usage."""
        if tool_id not in self.tools:
            raise ValueError(f"Tool {tool_id} not found")
        
        tool = self.tools[tool_id]
        
        import time
        start = time.time()
        
        try:
            result = tool.function(**kwargs)
            success = True
        except Exception as e:
            result = {"error": str(e)}
            success = False
        
        latency = time.time() - start
        tool.record_usage(latency, success)
        
        return result
    
    def update_skill(self, skill_id: str, success: bool) -> None:
        """Update skill after usage."""
        if skill_id in self.skills:
            self.skills[skill_id].update_usage(success)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        return {
            **self.stats,
            'memory_by_type': {
                mtype.value: len(ids) 
                for mtype, ids in self.type_index.items()
            },
            'top_tags': sorted(
                [(tag, len(ids)) for tag, ids in self.tag_index.items()],
                key=lambda x: x[1],
                reverse=True
            )[:10],
            'top_skills': sorted(
                [(s.skill_name, s.proficiency) for s in self.skills.values()],
                key=lambda x: x[1],
                reverse=True
            )[:10],
            'most_used_tools': sorted(
                [(t.tool_name, t.usage_count) for t in self.tools.values()],
                key=lambda x: x[1],
                reverse=True
            )[:10]
        }
    
    def export_to_json(self, filepath: str) -> None:
        """Export all data to JSON."""
        data = {
            'memories': [m.to_dict() for m in self.memories.values()],
            'skills': [s.to_dict() for s in self.skills.values()],
            'tools': [t.to_dict() for t in self.tools.values()],
            'statistics': self.get_statistics()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)


# Common utility tools that can be registered

def tool_encode_text(text: str, model: Optional[Any] = None) -> torch.Tensor:
    """Encode text to embedding."""
    # Mock implementation - replace with actual encoder
    return torch.randn(512)


def tool_extract_tags(text: str) -> Set[str]:
    """Extract tags from text."""
    # Simple keyword extraction
    words = text.lower().split()
    tags = {word for word in words if len(word) > 4}
    return tags


def tool_compute_similarity(emb1: torch.Tensor, emb2: torch.Tensor) -> float:
    """Compute cosine similarity between embeddings."""
    return F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0), dim=-1).item()


def tool_temporal_decay(created_at: datetime, decay_rate: float = 0.0001) -> float:
    """Compute temporal decay factor."""
    time_diff = (datetime.now() - created_at).total_seconds()
    return float(np.exp(-decay_rate * time_diff))


def tool_merge_embeddings(embeddings: List[torch.Tensor],
                         weights: Optional[List[float]] = None) -> torch.Tensor:
    """Merge multiple embeddings with optional weights."""
    if weights is None:
        weights = [1.0 / len(embeddings)] * len(embeddings)
    
    merged = sum(w * emb for w, emb in zip(weights, embeddings))
    return F.normalize(merged, dim=-1)


if __name__ == '__main__':
    print("="*70)
    print("UNIFIED TOOLS AND UTILITIES SYSTEM")
    print("="*70)
    
    # Create unified manager
    manager = UnifiedMemoryManager(embed_dim=512)
    
    # Register common tools
    print("\n[Step 1] Registering tools:")
    tool_ids = [
        manager.register_tool(
            "encode_text",
            "Encode text to embedding",
            tool_encode_text,
            {"text": "string"},
            "tensor"
        ),
        manager.register_tool(
            "extract_tags",
            "Extract tags from text",
            tool_extract_tags,
            {"text": "string"},
            "set"
        ),
        manager.register_tool(
            "compute_similarity",
            "Compute embedding similarity",
            tool_compute_similarity,
            {"emb1": "tensor", "emb2": "tensor"},
            "float"
        )
    ]
    print(f"  ✓ Registered {len(tool_ids)} tools")
    
    # Store memories
    print("\n[Step 2] Storing memories:")
    for i in range(10):
        memory_id = manager.store_memory(
            embedding=torch.randn(512),
            memory_type=MemoryType.EPISODIC,
            source="test_source",
            relevance=0.5 + i * 0.05,
            tags={f"tag_{i%3}", "test"}
        )
    print(f"  ✓ Stored 10 memories")
    
    # Register skills
    print("\n[Step 3] Registering skills:")
    skill_id = manager.register_skill(
        "code_generation",
        "programming",
        {"languages": ["python", "rust"]}
    )
    print(f"  ✓ Registered skill: {skill_id}")
    
    # Search memories
    print("\n[Step 4] Searching memories:")
    query = torch.randn(512)
    results = manager.recall_memory(
        query,
        filters={'tags': ['test'], 'min_relevance': 0.5},
        top_k=5
    )
    print(f"  ✓ Found {len(results)} relevant memories")
    
    # Use tool
    print("\n[Step 5] Using tools:")
    result = manager.use_tool(tool_ids[1], text="test sample text")
    print(f"  ✓ Tool result: {result}")
    
    # Get statistics
    print("\n[Step 6] Statistics:")
    stats = manager.get_statistics()
    print(f"  Total memories: {stats['total_memories']}")
    print(f"  Total skills: {stats['total_skills']}")
    print(f"  Total tools: {stats['total_tools']}")
    print(f"  Total searches: {stats['total_searches']}")
    print(f"  Avg search time: {stats['average_search_time']*1000:.2f}ms")
    
    print("\n" + "="*70)
    print("UNIFIED SYSTEM OPERATIONAL")
    print("="*70)
