export interface Document {
  id: string;
  title: string;
  type?: string;
  era?: string;
  region?: string;
  source?: string;
  url?: string;
  summary?: string;
  score?: number;
  component_scores?: {
    simrank: number;
    horn: number;
    embedding: number;
  };
  explanations?: string[];
  entities?: {
    locations?: string[];
    persons?: string[];
    organizations?: string[];
    dates?: string[];
  };
  keywords?: string[];
  cluster_label?: string;
  classifications?: {
    heritage_types?: string[];
    domains?: string[];
    time_period?: string;
    region?: string;
    architectural_styles?: string[];
  };
  word_count?: number;
}

export interface Entity {
  id: string;
  name: string;
  type: "locations" | "persons" | "organizations" | string;
  count: number;
}

export interface Collection {
  id: string;
  name: string;
  count: number;
  thumbnail?: string;
}

export interface SearchFilters {
  era?: string[];
  region?: string[];
  type?: string[];
  domain?: string[];
  query?: string;
}

export interface ParsedQuery {
  locations?: string[];
  persons?: string[];
  organizations?: string[];
  heritage_types?: string[];
  domains?: string[];
  region?: string | null;
  time_period?: string | null;
  architectural_styles?: string[];
}

export interface SearchResponse {
  documents: Document[];
  total: number;
  query: string;
  ensemble_method?: string;
  query_type?: string;
  parsed_query?: ParsedQuery;
}

export interface RecommendationResponse {
  documents: Document[];
  method: string;
}

export interface GraphNode {
  id: string;
  label: string;
  type: string;
}

export interface GraphEdge {
  source: string;
  target: string;
  weight: number;
  relation: string;
}

export interface KGNode {
  id: string;
  name: string;
  type: string;
  count: number;
  horn_weight: number;
  cluster_id?: number;
  cluster_label?: string;
  is_top_central: boolean;
}

export interface KGEdge {
  source: string;
  target: string;
  weight: number;
  edge_type: string;
}

export interface KGGraphResponse {
  nodes: KGNode[];
  edges: KGEdge[];
}

export interface KGNeighborsResponse {
  center: KGNode;
  neighbors: KGNode[];
  edges: KGEdge[];
}

export interface KGStats {
  total_nodes: number;
  total_edges: number;
  node_types: Record<string, number>;
  edge_types: Record<string, number>;
  density: number;
  average_degree: number;
}

export interface KnowledgeGraph {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

export interface SystemStats {
  total_documents: number;
  kg_nodes: number;
  kg_edges: number;
  node_types: Record<string, number>;
  era_counts?: Record<string, number>;
}

export interface Metrics {
  accuracy?: Record<string, number>;
  diversity?: Record<string, number>;
  heritage_specific?: Record<string, number>;
  efficiency?: Record<string, number>;
  [key: string]: unknown;
}
