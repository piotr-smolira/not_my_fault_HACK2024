import streamlit as st
from rdflib import Graph, URIRef
from streamlit_agraph import agraph, Node, Edge, Config
from urllib.parse import urlparse

def load_rdf_data(file):
    """Load RDF data from a file and return an RDFLib Graph."""
    g = Graph()
    g.parse(file, format='ttl')  # Adjust the format if needed
    return g

def extract_label(uri):
    """Extract a readable label from a URI."""
    parsed_uri = urlparse(uri)
    return parsed_uri.fragment or parsed_uri.path.split('/')[-1]

def get_node_info(g, node_uri):
    """Retrieve type, definition, and properties for a given node."""
    node_info = {
        'URI': node_uri,
        'Type': [],
        'Properties': []
    }
    for s, p, o in g.triples((URIRef(node_uri), None, None)):
        if p.endswith('#type'):
            node_info['Type'].append(o)
        else:
            node_info['Properties'].append((p, o))
    return node_info

def create_graph_nodes_edges(g):
    """Create graph nodes and edges for visualization."""
    nodes = {}
    edges = []
    
    for subj, pred, obj in g:
        subj_str, pred_str, obj_str = str(subj), str(pred), str(obj)
        
        subj_label = extract_label(subj_str)
        pred_label = extract_label(pred_str)
        obj_label = extract_label(obj_str)
        
        if subj_str not in nodes:
            nodes[subj_str] = Node(id=subj_str, label=subj_label)
        if obj_str not in nodes:
            nodes[obj_str] = Node(id=obj_str, label=obj_label)
        
        edges.append(Edge(source=subj_str, target=obj_str, label=pred_label))
    
    return list(nodes.values()), edges

# Set Streamlit app layout to wide
st.set_page_config(layout="wide")

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
    config = Config(width=900, height=600, directed=True, nodeHighlightBehavior=True, highlightColor="#F7A7A6", collapsible=True)
    return_value = agraph(nodes=nodes, edges=edges, config=config)

    # Display node information in the sidebar
    if return_value is not None:
        st.sidebar.write("Node Information")
        node_info = get_node_info(rdf_graph, return_value)
        
        st.sidebar.write(f"**URI:** {node_info['URI']}")
        st.sidebar.write("**Type:**")
        for node_type in node_info['Type']:
            st.sidebar.write(f"- {node_type}")
        
        st.sidebar.write("**Properties:**")
        for prop, val in node_info['Properties']:
            st.sidebar.write(f"- {prop}: {val}")

        # Show the detailed information for the node
        st.sidebar.write("### Detailed Information")
        for s, p, o in rdf_graph.triples((URIRef(return_value), None, None)):
            st.sidebar.write(f"{p}: {o}")
