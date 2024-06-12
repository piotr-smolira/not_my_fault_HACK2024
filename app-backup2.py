import streamlit as st
from rdflib import Graph
from streamlit_agraph import agraph, Node, Edge, Config

def load_rdf_data(file):
    """Load RDF data from a file and return an RDFLib Graph."""
    g = Graph()
    g.parse(file, format='ttl')  # Adjust the format if needed
    return g

def create_graph_nodes_edges(g):
    """Create graph nodes and edges for visualization."""
    nodes = {}
    edges = []
    
    for subj, pred, obj in g:
        subj_str, pred_str, obj_str = str(subj), str(pred), str(obj)
        
        if subj_str not in nodes:
            nodes[subj_str] = Node(id=subj_str, label=subj_str)
        if obj_str not in nodes:
            nodes[obj_str] = Node(id=obj_str, label=obj_str)
        
        edges.append(Edge(source=subj_str, target=obj_str, label=pred_str))
    
    return list(nodes.values()), edges

# Streamlit app title
st.title("RDF Data Visualizer")

# File uploader widget
uploaded_file = st.file_uploader("Choose an RDF file", type="ttl")

if uploaded_file is not None:
    st.write("File uploaded successfully.")
    
    # Load RDF data
    rdf_graph = load_rdf_data(uploaded_file)
    
    # Create nodes and edges for graph visualization
    nodes, edges = create_graph_nodes_edges(rdf_graph)
    
    # Visualize the graph using streamlit-agraph
    config = Config(width=800, height=600, directed=True, nodeHighlightBehavior=True, highlightColor="#F7A7A6", collapsible=True)
    return_value = agraph(nodes=nodes, edges=edges, config=config)
