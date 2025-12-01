# Inventory Management Assistant (Google ADK)

An AI-driven inventory advisor powered by Google’s Agent Development Kit (ADK), Gemini models, and a Gradio UI. The agent consults a SQLite database to deliver reorder guidance, stock summaries, and proactive alerts.

## Features

- **Google ADK Agent** using `Agent`, `Runner`, and Gemini (`gemini-2.5-flash`) for LLM reasoning.
- **Tool-augmented responses** via three function tools:
  - `get_inventory_data(product)` – retrieve stock, demand, and lead time.
  - `get_all_inventory()` – list all products with contextual status.
  - `calculate_reorder_quantity(product)` – recommend reorder quantities when stock is low.
- **Security layers**: prompt-injection guardrails, role-switch prevention, system prompt protection, and input-length checks.
- **Performance helpers**: TTL-based input cache with hit/miss stats, database initialization guard, and environment validation.
- **User experience**: Gradio chat app with quick examples, cache dashboards, and optional email notifications.

## Setup

1. **Environment**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. **Configuration**
   Create `.env`:
   ```
   GOOGLE_API_KEY=your-google-key
   EMAIL_HOST=smtp.example.com
   EMAIL_PORT=587
   EMAIL_USER=alerts@example.com
   EMAIL_PASS=app-password
   EMAIL_TO=ops@example.com
   ```
3. **Database**
   Ensure `database.py` exposes `init_database()` and `get_db_connection()` returning an `inventory` table with `product_name`, `current_stock`, `average_demand`, `lead_time`.

4. **Run**
   ```powershell
   python inventoryplanner_adk.py
   ```
   Gradio launches at the printed URL.

## Architecture Overview

- **Guardrails** (`Guardrails.evaluate`) filter hostile inputs before the agent runs.
- **Caching** (`InputCache`) hashes normalized prompts for quicker repeat replies.
- **Agent Loop**
  1. Validate input.
  2. Serve from cache when possible.
  3. Invoke ADK `Runner` with the active session.
  4. Persist response in cache; surface to UI.
- **Email Utility** (`send_email`) optionally forwards summaries to an ops mailbox.

## Operations

- Use the “🔄 Refresh Cache Stats” button to inspect cache health.
- “🗑️ Clear Cache” purges stored prompts/responses.
- Chat history is maintained within the Gradio session; refresh the browser to reset.

## Extending the Agent

- Add new SQLite-backed helpers and register them via `tools=[...]`.
- Swap `gemini-2.5-flash` for other Gemini variants by updating the agent `model`.
- Integrate telemetry by decorating `inventory_chat` or tool functions with additional tracing.

## Troubleshooting

| Issue | Resolution |
| --- | --- |
| `GOOGLE_API_KEY is not set` | Confirm `.env` is loaded or export the key in shell. |
| `RuntimeError: Failed to initialize database` | Verify schema creation inside `init_database()` and file permissions. |
| Gradio UI blank | Ensure dependencies installed and port not blocked. |
| Slow responses | Increase cache TTL/size or upgrade Gemini model tier. |

Enjoy streamlined inventory decisions with Gemini and Google ADK.
