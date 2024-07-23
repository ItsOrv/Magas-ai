# Magas-AI Telegram Bot

This repository contains the code for a Telegram bot named "Magas". The bot leverages NLP models for natural language understanding and response generation. It is designed to interact with users, collect their interests, search relevant scientific abstracts, and generate coherent responses based on user input and interests.

## Features

- **User Interaction**: The bot interacts with users, collects their messages, and stores their interests.
- **NLP Integration**: Uses the `sentence-transformers/LaBSE` model for semantic similarity and `MehdiHosseiniMoghadam/AVA-Llama-3-V2` for response generation.
- **Corpus Search**: Searches through a corpus of scientific abstracts to find relevant information based on user queries.
- **Response Generation**: Generates contextually appropriate responses using user interests and the retrieved corpus information.

## Setup

### Prerequisites

- Python 3.8+
- Telegram Bot Token
- GPU (A100 recommended)

### Installation

1. Clone the repository:
    ```
    git clone https://github.com/itsorv/magas-ai.git
    cd magas-ai
    ```

2. Create a virtual environment and activate it:
   linux
    ```
    python -m venv venv
    source venv/bin/activate
    ```
     Windows:
   ```
    python -m venv venv
    venv\Scripts\activate
   ```

4. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

5. replace your `data.csv` in the directory of the project.

6. Add your Telegram Bot Token to the code:
    ```
    TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"
    ```

### Running the Bot

Start the bot with:
```
python bot.py
```
Usage
/start: interaction with the bot.
Private messages: The bot will handle private messages by collecting user info and responding contextually.
Group messages: The bot will respond to group messages if they contain specific keywords.
Code Structure
bot.py: The main script containing the bot logic.
data.csv: The CSV file containing scientific abstracts.
user_data.json: A JSON file to store user information and chat history.

To Do List
Improve Error Handling:

Implement comprehensive try-except blocks around critical sections of the code.
Enhance logging to include more context-specific information.
Optimize Model Loading:

Consider lazy loading or preloading models to reduce memory footprint and increase efficiency.
Enhance User Data Collection:

Implement more sophisticated methods for extracting user interests from messages.
Add mechanisms to periodically clean and update user data.
Expand Corpus:

Allow dynamic updates to the corpus by adding new abstracts without restarting the bot.
Implement a mechanism to fetch and integrate external datasets.
Refine Response Generation:

Improve the prompt structure for more coherent and relevant responses.
Experiment with different model parameters (e.g., max_length, temperature) to optimize response quality.
User Interface:

Develop a more interactive user interface for better engagement.
Implement multimedia message support (images, audio, etc.).
Testing and Validation:

Add unit tests and integration tests to ensure code reliability and stability.
Set up continuous integration (CI) for automated testing and deployment.
Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

