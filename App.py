import Graphdb as gd
import streamlit as st
from pyvis.network import Network
import networkx as nx
import tempfile
import os
st.set_page_config(page_title="WikiGraph Explorer", layout="centered")


# ✅ Ensure session state variables exist
if "graph_created" not in st.session_state:
    st.session_state.graph_created = False
if "current_entity" not in st.session_state:
    st.session_state.current_entity = None
if "vector_index" not in st.session_state:
    st.session_state.vector_index = None
if "show_graph" not in st.session_state:
    st.session_state.show_graph = False  # ✅ Toggle for showing/hiding the graph
if "graph_html" not in st.session_state:
    st.session_state.graph_html = None  # ✅ Cached graph visualization

def reset_graph():
    """Deletes everything in the existing graph (nodes, relationships, indexes, constraints)."""
    try:
        graph = gd.Neo4jGraph()
        graph.query("MATCH (n) DETACH DELETE n")

        constraints = graph.query("SHOW CONSTRAINTS YIELD name RETURN name")
        for constraint in constraints:
            graph.query(f"DROP CONSTRAINT {constraint['name']}")
        indexes = graph.query("SHOW INDEXES YIELD name RETURN name")
        for index in indexes:
            graph.query(f"DROP INDEX {index['name']}")

        # main_window.write("Graph reset successfully!")
        st.session_state.graph_created = False
        st.session_state.vector_index = None
        st.session_state.graph_html = None  # ✅ Clear cached visualization

    except Exception as e:
        st.write(f"Error deleting graph: {e}")

def visualize_graph():
    """Creates an interactive visualization of the Neo4j graph."""
    graph = gd.Neo4jGraph()
    query = "MATCH (n)-[r]->(m) RETURN n.id AS source, type(r) AS relationship, m.id AS target"
    results = graph.query(query)

    # ✅ Add nodes to Pyvis graph
    G = nx.DiGraph()
    for record in results:
        G.add_edge(record["source"], record["target"], label=record["relationship"])

    net = Network(height="600px", width="100%", directed=True, notebook=False)
    net.from_nx(G)
    net.toggle_physics(True)  # ✅ Enable physics for smooth movement
    # ✅ Save as an HTML file for embedding in Streamlit
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
        net.save_graph(tmp_file.name)
        return tmp_file.name

if __name__ == "__main__":

    # ✅ Set page configuration

    # ✅ Custom CSS for a cleaner look
    st.markdown(
        """
        <style>
            /* Center everything */
            .stTextInput>div>div>input {
                text-align: center;
                font-size: 18px;
            }
            /* Title styling */
            h1 {
                color: #FFA500;
                text-align: center;
                font-weight: bold;
            }
            /* Subtitle */
            .subtitle {
                text-align: center;
                font-size: 16px;
                color: #BBBBBB;
                margin-bottom: 20px;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<h1>🔍 WikiGraph Explorer</h1>", unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Explore relationships and entities with Graph-based Retrieval</p>',
                unsafe_allow_html=True)

    s = st.text_input("Enter an Entity (e.g., a person or organization)")
    main_window = st.empty()
    st.divider()

    if s:
        if st.session_state.current_entity != s:
            main_window.write(f"Creating Graph for '{s}'...")
            reset_graph()
            docs = gd.load_data(s)
            gd.create_graph(docs)

            st.session_state.vector_index = gd.Neo4jVector.from_existing_graph(
                gd.encoder,
                search_type='hybrid',
                node_label='Document',
                text_node_properties=['text'],
                embedding_node_property='embedding'
            )

            st.session_state.graph_created = True
            st.session_state.current_entity = s
            main_window.write("Graph created successfully!")

    # ✅ Checkbox to show/hide graph
        if st.session_state.graph_created:
            if st.session_state.graph_created:
                if st.button("Toggle Graph View"):
                    st.session_state.show_graph = not st.session_state.show_graph

                if st.session_state.show_graph:
                    graph_html = visualize_graph()
                    st.components.v1.html(open(graph_html, "r").read(), height=600)

            question = st.text_input("Enter your query")
            if question:
                if st.session_state.vector_index is None:
                    st.write("Error: Vector index is not initialized!")
                else:
                    response,context = gd.answer_query(question, st.session_state.vector_index)
                    st.subheader("Response:")
                    st.write(response)
                    # ✅ Split structured and unstructured data
                    structured_part = context.split("\n\nUnstructured Data:\n")
                    st.subheader("Context Used:")
                    st.write(structured_part)

