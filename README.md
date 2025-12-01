# **Inventory Management Agent**

This project is an AI-powered inventory management assistant built with Google’s Agent Development Kit (ADK), Gemini models, and a Gradio user interface. It leverages a Retrieval-Augmented Generation (RAG) architecture to answer natural language queries about inventory by querying a real-time database.

**Core Technology Stack**

- Google ADK with `Gemini` via `LlmAgent`/`Runner`
- Gradio front end
- SQLite inventory database
- Custom guardrails and caching utilities

**Tooling**

The ADK agent exposes:
- **get_inventory_data** – fetch stock, demand, lead time for a product.
- **get_all_inventory** – summarize all inventory items.
- **calculate_reorder_quantity** – recommend reorder actions based on demand and lead time.
