# FastAPI Agent API

This is a FastAPI application that exposes an API for running a LangGraph-based AI agent and managing its run history.

## Setup

1.  **Install Dependencies:**

    ```bash
    pip install -r fastapi_app/requirements.txt
    pip install -r langgraph_stance_analyzer/requirements.txt
    ```

2.  **Ollama:** Ensure you have Ollama running and the `llama3.1:8b` model pulled. You can pull the model using:

    ```bash
    ollama pull llama3.1:8b
    ```

## Running the Application

To start the FastAPI server, navigate to the project root directory and run:

```bash
cd fastapi_app
uvicorn main:app --host 0.0.0.0 --port 8000
```

The API will be accessible at `http://0.0.0.0:8000`.

## API Endpoints

### 1. Run Agent

-   **Endpoint:** `/run_agent`
-   **Method:** `POST`
-   **Description:** Initiates an agent run with the provided text and stores the result.
-   **Request Body (JSON):**

    ```json
    {
        "text": "Your input text for the agent"
    }
    ```

-   **Response (JSON):**

    ```json
    {
        "run_id": "<uuid>",
        "status": "completed" | "failed",
        "input_text": "Your input text for the agent",
        "result": { ... }, // Agent's output or error details
        "timestamp": "<ISO 8601 datetime>"
    }
    ```

### 2. Get All Agent Runs

-   **Endpoint:** `/agent_runs`
-   **Method:** `GET`
-   **Description:** Retrieves a list of all previous agent runs.
-   **Response (JSON Array):** A list of objects, each matching the `Run Agent` response structure.

### 3. Get Specific Agent Run

-   **Endpoint:** `/agent_runs/{run_id}`
-   **Method:** `GET`
-   **Description:** Retrieves the details of a specific agent run by its ID.
-   **Response (JSON):** An object matching the `Run Agent` response structure.

## History Storage

All agent run history is stored locally in the `agent_runs/` directory at the project root. Each successful or failed agent run generates a unique JSON file named after its `run_id` (e.g., `agent_runs/<uuid>.json`). These files contain the input text, the agent's output (or error details), the status of the run, and a timestamp.
