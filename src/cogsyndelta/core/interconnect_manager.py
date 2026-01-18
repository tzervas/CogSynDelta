"""
Intelligent Communication Interconnect Manager

Specialized submodel/section for managing mHC pathways and context.

This acts as the "corpus callosum" of the synthetic brain, intelligently:
1. Routing information between brain regions/sections
2. Managing communication bandwidth and priorities
3. Maintaining contextual coherence across sections
4. Optimizing information flow based on task demands
5. Learning optimal routing patterns over time

Key features:
- Attention-based routing (learns which sections need to communicate)
- Context summarization and propagation
- Bandwidth allocation and congestion control
- Dynamic pathway strength adjustment
- Multi-hop routing for distant sections
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
import numpy as np


@dataclass
class CommunicationContext:
    """Contextual information for inter-section communication."""
    source_section: str
    target_section: str
    context_embedding: torch.Tensor
    importance: float
    timestamp: float
    pathway_strength: float
    latency: float
    bandwidth_used: int
    
    def to_dict(self) -> Dict:
        return {
            'source': self.source_section,
            'target': self.target_section,
            'importance': self.importance,
            'pathway_strength': self.pathway_strength,
            'latency': self.latency,
            'bandwidth': self.bandwidth_used
        }


@dataclass
class RoutingDecision:
    """Routing decision made by the interconnect manager."""
    route_path: List[str]
    total_cost: float
    estimated_latency: float
    bandwidth_required: int
    priority: float


class ContextualAttentionRouter(nn.Module):
    """
    Attention-based router for determining which sections should communicate.
    
    Uses transformer-style attention to learn:
    - Which brain regions need to exchange information
    - How much bandwidth to allocate
    - What context is relevant to share
    """
    
    def __init__(self, embed_dim: int = 512, num_heads: int = 8, num_sections: int = 10) -> None:
        super(ContextualAttentionRouter, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_sections = num_sections
        
        # Multi-head attention for routing decisions
        self.routing_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Importance estimator
        self.importance_network = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid()
        )
        
        # Bandwidth allocator
        self.bandwidth_allocator = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1),
            nn.Softplus()  # Always positive bandwidth
        )
        
        # Context summarizer (compresses context for efficient transmission)
        self.context_summarizer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim // 4)
        )
        
        # Context expander (decompresses at destination)
        self.context_expander = nn.Sequential(
            nn.Linear(embed_dim // 4, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim)
        )
    
    def forward(self, section_states: Dict[str, torch.Tensor],
                section_ids: List[str]) -> Dict[Tuple[str, str], float]:
        """
        Compute routing decisions for all section pairs.
        
        Args:
            section_states: Current state of each section {section_id: state_tensor}
            section_ids: List of active section IDs
            
        Returns:
            Dictionary of (source, target) -> communication_importance
        """
        # Stack section states
        states_list = [section_states[sid] for sid in section_ids]
        states = torch.stack(states_list)  # [num_sections, embed_dim]
        
        # Compute attention matrix (who should talk to whom)
        attn_output, attn_weights = self.routing_attention(
            states.unsqueeze(0),
            states.unsqueeze(0),
            states.unsqueeze(0)
        )
        
        # Extract pairwise communication importance
        attn_weights = attn_weights.squeeze(0)  # [num_sections, num_sections]
        
        routing_decisions = {}
        for i, source_id in enumerate(section_ids):
            for j, target_id in enumerate(section_ids):
                if i != j:  # No self-communication
                    importance = attn_weights[i, j].item()
                    routing_decisions[(source_id, target_id)] = importance
        
        return routing_decisions
    
    def compute_importance(self, source_state: torch.Tensor,
                          target_state: torch.Tensor) -> float:
        """Compute importance of communication between two sections."""
        combined = torch.cat([source_state, target_state], dim=-1)
        importance = self.importance_network(combined)
        return importance.item()
    
    def allocate_bandwidth(self, source_state: torch.Tensor,
                          target_state: torch.Tensor) -> int:
        """Determine bandwidth allocation for communication."""
        combined = torch.cat([source_state, target_state], dim=-1)
        bandwidth = self.bandwidth_allocator(combined)
        return int(bandwidth.item() * 100)  # Scale to reasonable range
    
    def compress_context(self, context: torch.Tensor) -> torch.Tensor:
        """Compress context for efficient transmission."""
        return self.context_summarizer(context)
    
    def decompress_context(self, compressed: torch.Tensor) -> torch.Tensor:
        """Decompress context at destination."""
        return self.context_expander(compressed)


class PathwayOptimizer(nn.Module):
    """
    Learns optimal pathway strengths through experience.
    
    Uses reinforcement learning principles to adjust mHC pathway strengths
    based on successful vs unsuccessful communication patterns.
    """
    
    def __init__(self, embed_dim: int = 512) -> None:
        super(PathwayOptimizer, self).__init__()
        
        self.embed_dim = embed_dim
        
        # Value network (estimates value of a pathway configuration)
        self.value_network = nn.Sequential(
            nn.Linear(embed_dim * 2 + 2, embed_dim),  # +2 for current_strength, usage_history
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1)
        )
        
        # Policy network (proposes pathway strength adjustments)
        self.policy_network = nn.Sequential(
            nn.Linear(embed_dim * 2 + 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1),
            nn.Tanh()  # Output in [-1, 1] for strength adjustment
        )
    
    def evaluate_pathway(self, source_state: torch.Tensor,
                        target_state: torch.Tensor,
                        current_strength: float,
                        usage_history: float) -> float:
        """Evaluate current pathway configuration."""
        features = torch.cat([
            source_state,
            target_state,
            torch.tensor([current_strength]),
            torch.tensor([usage_history])
        ], dim=-1)
        
        value = self.value_network(features)
        return value.item()
    
    def propose_adjustment(self, source_state: torch.Tensor,
                          target_state: torch.Tensor,
                          current_strength: float,
                          usage_history: float) -> float:
        """Propose strength adjustment for pathway."""
        features = torch.cat([
            source_state,
            target_state,
            torch.tensor([current_strength]),
            torch.tensor([usage_history])
        ], dim=-1)
        
        adjustment = self.policy_network(features)
        return adjustment.item() * 0.1  # Scale to small adjustments


class ContextPropagationEngine(nn.Module):
    """
    Manages context propagation across the network.
    
    Ensures coherent context is maintained across distant sections,
    handling multi-hop routing and context accumulation.
    """
    
    def __init__(self, embed_dim: int = 512, max_hops: int = 3) -> None:
        super(ContextPropagationEngine, self).__init__()
        
        self.embed_dim = embed_dim
        self.max_hops = max_hops
        
        # Context integration networks (for each hop)
        self.hop_integrators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim)
            ) for _ in range(max_hops)
        ])
        
        # Context decay factor (information degrades over hops)
        self.decay_factors = nn.Parameter(torch.ones(max_hops) * 0.9)
        
        # Relevance gating (determines what context is relevant)
        self.relevance_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid()
        )
    
    def propagate_context(self, source_context: torch.Tensor,
                         intermediate_states: List[torch.Tensor],
                         target_state: torch.Tensor) -> torch.Tensor:
        """
        Propagate context from source to target through intermediate sections.
        
        Args:
            source_context: Original context from source
            intermediate_states: States of intermediate sections
            target_state: State of final target section
            
        Returns:
            Propagated and integrated context
        """
        current_context = source_context
        
        # Propagate through intermediate hops
        for hop_idx, intermediate_state in enumerate(intermediate_states[:self.max_hops]):
            # Integrate with intermediate state
            combined = torch.cat([current_context, intermediate_state], dim=-1)
            integrated = self.hop_integrators[hop_idx](combined)
            
            # Apply decay
            current_context = integrated * self.decay_factors[hop_idx]
        
        # Final integration with target
        combined = torch.cat([current_context, target_state], dim=-1)
        relevance_mask = self.relevance_gate(combined)
        
        final_context = current_context * relevance_mask
        
        return final_context


class CongestionController:
    """
    Manages bandwidth and prevents communication congestion.
    
    Implements:
    - Priority-based queuing
    - Flow control
    - Backpressure signaling
    - Dynamic bandwidth allocation
    """
    
    def __init__(self, total_bandwidth: int = 10000) -> None:
        self.total_bandwidth = total_bandwidth
        self.allocated_bandwidth: Dict[Tuple[str, str], int] = {}
        self.message_queue: Dict[int, deque] = defaultdict(deque)  # priority -> queue
        self.bandwidth_usage_history: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    
    def allocate(self, pathway: Tuple[str, str], required_bandwidth: int,
                priority: int = 5) -> bool:
        """
        Attempt to allocate bandwidth for a pathway.
        
        Args:
            pathway: (source, target) tuple
            required_bandwidth: Bandwidth required
            priority: Priority level (0-10, higher is more important)
            
        Returns:
            True if allocation successful, False if congested
        """
        current_usage = sum(self.allocated_bandwidth.values())
        
        if current_usage + required_bandwidth <= self.total_bandwidth:
            self.allocated_bandwidth[pathway] = required_bandwidth
            self.bandwidth_usage_history[pathway].append(required_bandwidth)
            return True
        
        # Try to preempt lower priority allocations
        if priority >= 7:  # High priority
            # Find lowest priority allocation
            sorted_allocs = sorted(
                self.allocated_bandwidth.items(),
                key=lambda x: self._get_priority(x[0])
            )
            
            for low_priority_pathway, bw in sorted_allocs:
                if self._get_priority(low_priority_pathway) < priority:
                    # Preempt
                    del self.allocated_bandwidth[low_priority_pathway]
                    self.allocated_bandwidth[pathway] = required_bandwidth
                    return True
        
        # Queue the message
        self.message_queue[priority].append((pathway, required_bandwidth))
        return False
    
    def release(self, pathway: Tuple[str, str]) -> None:
        """Release bandwidth allocation."""
        if pathway in self.allocated_bandwidth:
            del self.allocated_bandwidth[pathway]
            
            # Process queued messages
            self._process_queue()
    
    def _process_queue(self) -> None:
        """Process queued messages in priority order."""
        for priority in sorted(self.message_queue.keys(), reverse=True):
            while self.message_queue[priority]:
                pathway, bandwidth = self.message_queue[priority][0]
                
                if self.allocate(pathway, bandwidth, priority):
                    self.message_queue[priority].popleft()
                else:
                    break  # Can't allocate, wait for more bandwidth
    
    def _get_priority(self, pathway: Tuple[str, str]) -> int:
        """Get priority of a pathway (based on usage history)."""
        history = self.bandwidth_usage_history.get(pathway, [])
        if not history:
            return 5  # Default priority
        
        # Higher usage = higher priority
        avg_usage = np.mean(history[-10:])
        return int(min(10, avg_usage / 100))
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get congestion statistics."""
        return {
            'total_bandwidth': self.total_bandwidth,
            'used_bandwidth': sum(self.allocated_bandwidth.values()),
            'utilization': sum(self.allocated_bandwidth.values()) / self.total_bandwidth,
            'active_pathways': len(self.allocated_bandwidth),
            'queued_messages': sum(len(q) for q in self.message_queue.values())
        }


class IntelligentInterconnectManager(nn.Module):
    """
    Main interconnect management system.
    
    Coordinates all aspects of inter-section communication:
    - Routing decisions (attention-based)
    - Pathway optimization (learned)
    - Context propagation (multi-hop)
    - Congestion control (bandwidth management)
    - Performance monitoring
    """
    
    def __init__(self, embed_dim: int = 512, num_sections: int = 10,
                 total_bandwidth: int = 10000) -> None:
        super(IntelligentInterconnectManager, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_sections = num_sections
        
        # Core components
        self.router = ContextualAttentionRouter(embed_dim, num_heads=8, num_sections=num_sections)
        self.optimizer = PathwayOptimizer(embed_dim)
        self.propagator = ContextPropagationEngine(embed_dim, max_hops=3)
        self.congestion_controller = CongestionController(total_bandwidth)
        
        # Pathway registry
        self.pathways: Dict[Tuple[str, str], Dict[str, Any]] = {}
        
        # Performance tracking
        self.communication_history: List[CommunicationContext] = []
        self.performance_metrics: Dict[str, List[float]] = defaultdict(list)
    
    def register_pathway(self, source: str, target: str,
                        initial_strength: float = 1.0) -> Any:
        """Register a communication pathway."""
        self.pathways[(source, target)] = {
            'strength': initial_strength,
            'usage_count': 0,
            'total_latency': 0.0,
            'success_rate': 1.0
        }
    
    def route_communication(self, source_id: str, target_id: str,
                          source_state: torch.Tensor, target_state: torch.Tensor,
                          message: torch.Tensor,
                          all_section_states: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, RoutingDecision]:
        """
        Intelligently route communication from source to target.
        
        Args:
            source_id: Source section ID
            target_id: Target section ID
            source_state: Current state of source
            target_state: Current state of target
            message: Message to transmit
            all_section_states: All section states for multi-hop routing
            
        Returns:
            (transmitted_message, routing_decision)
        """
        pathway_key = (source_id, target_id)
        
        # Check if direct pathway exists
        if pathway_key in self.pathways:
            # Direct routing
            importance = self.router.compute_importance(source_state, target_state)
            bandwidth = self.router.allocate_bandwidth(source_state, target_state)
            
            # Attempt bandwidth allocation
            if self.congestion_controller.allocate(pathway_key, bandwidth, priority=int(importance * 10)):
                # Transmit with compression
                compressed_message = self.router.compress_context(message)
                transmitted = compressed_message * self.pathways[pathway_key]['strength']
                
                # Update pathway stats
                self.pathways[pathway_key]['usage_count'] += 1
                
                # Decompress at destination
                final_message = self.router.decompress_context(transmitted)
                
                # Release bandwidth
                self.congestion_controller.release(pathway_key)
                
                routing_decision = RoutingDecision(
                    route_path=[source_id, target_id],
                    total_cost=1.0,
                    estimated_latency=1.0,
                    bandwidth_required=bandwidth,
                    priority=importance
                )
                
                # Record communication
                self._record_communication(source_id, target_id, message, importance, 1.0)
                
                return final_message, routing_decision
            else:
                # Congestion detected
                return torch.zeros_like(message), RoutingDecision(
                    route_path=[],
                    total_cost=float('inf'),
                    estimated_latency=float('inf'),
                    bandwidth_required=bandwidth,
                    priority=importance
                )
        
        else:
            # Multi-hop routing required
            route_path = self._find_route(source_id, target_id, all_section_states)
            
            if not route_path:
                return torch.zeros_like(message), RoutingDecision(
                    route_path=[],
                    total_cost=float('inf'),
                    estimated_latency=float('inf'),
                    bandwidth_required=0,
                    priority=0.0
                )
            
            # Propagate through intermediate sections
            intermediate_states = [all_section_states[sid] for sid in route_path[1:-1]]
            propagated_message = self.propagator.propagate_context(
                message, intermediate_states, target_state
            )
            
            routing_decision = RoutingDecision(
                route_path=route_path,
                total_cost=len(route_path),
                estimated_latency=len(route_path) * 1.5,
                bandwidth_required=100 * len(route_path),
                priority=0.5
            )
            
            return propagated_message, routing_decision
    
    def _find_route(self, source: str, target: str,
                   all_states: Dict[str, torch.Tensor]) -> List[str]:
        """Find multi-hop route using pathways."""
        # Simple BFS to find route
        queue = deque([(source, [source])])
        visited = {source}
        
        while queue:
            current, path = queue.popleft()
            
            if current == target:
                return path
            
            # Find outgoing pathways
            for (src, tgt) in self.pathways.keys():
                if src == current and tgt not in visited:
                    visited.add(tgt)
                    queue.append((tgt, path + [tgt]))
        
        return []  # No route found
    
    def optimize_pathways(self, section_states: Dict[str, torch.Tensor]) -> None:
        """Optimize all pathway strengths based on learned patterns."""
        for (source, target), pathway_info in self.pathways.items():
            if source in section_states and target in section_states:
                source_state = section_states[source]
                target_state = section_states[target]
                
                usage_history = pathway_info['usage_count'] / 100.0
                current_strength = pathway_info['strength']
                
                # Propose adjustment
                adjustment = self.optimizer.propose_adjustment(
                    source_state, target_state,
                    current_strength, usage_history
                )
                
                # Apply adjustment with bounds
                new_strength = max(0.1, min(2.0, current_strength + adjustment))
                pathway_info['strength'] = new_strength
    
    def _record_communication(self, source: str, target: str,
                            message: torch.Tensor, importance: float,
                            latency: float) -> Any:
        """Record communication for analysis."""
        context = CommunicationContext(
            source_section=source,
            target_section=target,
            context_embedding=message.detach(),
            importance=importance,
            timestamp=0.0,  # Would use actual time
            pathway_strength=self.pathways.get((source, target), {}).get('strength', 1.0),
            latency=latency,
            bandwidth_used=message.numel() * 4
        )
        
        self.communication_history.append(context)
        
        # Track metrics
        self.performance_metrics['latency'].append(latency)
        self.performance_metrics['importance'].append(importance)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        congestion_stats = self.congestion_controller.get_statistics()
        
        return {
            'total_pathways': len(self.pathways),
            'total_communications': len(self.communication_history),
            'average_latency': np.mean(self.performance_metrics['latency']) if self.performance_metrics['latency'] else 0.0,
            'average_importance': np.mean(self.performance_metrics['importance']) if self.performance_metrics['importance'] else 0.0,
            'congestion': congestion_stats,
            'pathway_usage': {
                f"{src}->{tgt}": info['usage_count']
                for (src, tgt), info in self.pathways.items()
            }
        }


if __name__ == '__main__':
    print("="*70)
    print("INTELLIGENT COMMUNICATION INTERCONNECT MANAGER")
    print("="*70)
    
    # Create interconnect manager
    manager = IntelligentInterconnectManager(
        embed_dim=512,
        num_sections=5,
        total_bandwidth=10000
    )
    
    # Register pathways
    print("\n[Step 1] Registering pathways:")
    manager.register_pathway("visual", "prefrontal", initial_strength=0.9)
    manager.register_pathway("auditory", "prefrontal", initial_strength=0.8)
    manager.register_pathway("prefrontal", "motor", initial_strength=1.0)
    manager.register_pathway("prefrontal", "memory", initial_strength=0.95)
    print("  ✓ Registered 4 pathways")
    
    # Simulate section states
    print("\n[Step 2] Simulating communication:")
    section_states = {
        "visual": torch.randn(1, 512),
        "auditory": torch.randn(1, 512),
        "prefrontal": torch.randn(1, 512),
        "motor": torch.randn(1, 512),
        "memory": torch.randn(1, 512)
    }
    
    # Route communication
    message = torch.randn(1, 512)
    transmitted, routing = manager.route_communication(
        "visual", "prefrontal",
        section_states["visual"], section_states["prefrontal"],
        message, section_states
    )
    
    print(f"  ✓ Routed message from visual to prefrontal")
    print(f"    Route: {' -> '.join(routing.route_path)}")
    print(f"    Latency: {routing.estimated_latency:.2f}")
    print(f"    Bandwidth: {routing.bandwidth_required} units")
    
    # Optimize pathways
    print("\n[Step 3] Optimizing pathways:")
    manager.optimize_pathways(section_states)
    print("  ✓ Pathway strengths adjusted based on usage")
    
    # Get statistics
    print("\n[Step 4] Statistics:")
    stats = manager.get_statistics()
    print(f"  Total pathways: {stats['total_pathways']}")
    print(f"  Total communications: {stats['total_communications']}")
    print(f"  Bandwidth utilization: {stats['congestion']['utilization']:.1%}")
    
    print("\n" + "="*70)
    print("INTERCONNECT MANAGER OPERATIONAL")
    print("="*70)
