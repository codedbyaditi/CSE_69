# Intelligent Substation Bot

This project is an AI-powered substation assistant that combines real-time asset monitoring with an intelligent chatbot for maintenance support.

## Features

- **Real-time Dashboard**: Monitors Voltage, Current, and Temperature with simulated data.
- **Intelligent Chatbot**: Answers questions about equipment maintenance, tools, and safety precautions using a BERT-like logic.
- **Split-Screen UI**: View asset health and chat with the bot simultaneously.

## Structure

- `backend/`: Contains the FastAPI application and logic.
    - `app/`: Main application code (routes, static files, templates).
    - `models/`: AI models and knowledge base.
- `run.py`: Entry point script.

## Setup & Running

1.  **Install Dependencies**:
    ```bash
    pip install -r backend/requirements.txt
    ```

2.  **Run the Application**:
    ```bash
    python run.py
    ```

3.  **Access the Dashboard**:
    Open your browser to [http://localhost:8080](http://localhost:8080).
