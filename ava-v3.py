from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from sentence_transformers import SentenceTransformer, util
from telegram import Update, Bot
import pandas as pd
import logging
import torch
import json
import os

# logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

# models
device = "cpu"
model1 = SentenceTransformer('sentence-transformers/LaBSE')
model_name_or_id = "MehdiHosseiniMoghadam/AVA-Llama-3-V2"
try:
    model = AutoModelForCausalLM.from_pretrained(model_name_or_id, torch_dtype=torch.bfloat16, device=device)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_id)
    print("Model loaded on device:", model.device)
except ImportError as e:
    logging.error(f"ImportError: {e}")
    logging.error("Ensure `accelerate` and `bitsandbytes` are installed. Run: pip install accelerate bitsandbytes")
    raise

# csv data
try:
    paper_text = pd.read_csv('data.csv')['Abstract']
    paper_text = paper_text.dropna().reset_index(drop=True)
except FileNotFoundError as e:
    logging.error(f"Error reading data.csv: {e}")
    raise
corpus_embeddings = model1.encode(paper_text, show_progress_bar=True, convert_to_tensor=True)

# user data
user_data_file = "user_data.json"
if os.path.exists(user_data_file):
    with open(user_data_file, "r") as file:
        user_data = json.load(file)
else:
    user_data = {}

# save user data
def save_user_data():
    with open(user_data_file, "w") as file:
        json.dump(user_data, file)

# update user data
def collect_user_info(user_id, message):
    user = user_data.get(user_id, {"name": "", "age": "", "interests": [], "chat_history": []})
    user["interests"].extend(extract_interests(message))
    user_data[user_id] = user
    user_data[user_id]["chat_history"].append(message)
    save_user_data()

# interests
def extract_interests(message):
    interests = ["interest1", "interest2", "interest3"]
    return interests

# search the corpus
def search(inp_question, num_res, corpus_embeddings, paper_text):
    res = {}
    question_embedding = model1.encode(inp_question, convert_to_tensor=True)
    hits = util.semantic_search(question_embedding, corpus_embeddings, top_k=num_res)[0]
    for hit in hits:
        res[hit['corpus_id']] = paper_text[hit['corpus_id']]
    return pd.DataFrame(list(res.items()), columns=['id', 'res'])

# generate a response
def generate_response(prompt, user_interests):
    try:
        user_interest_text = "User interests: " + ", ".join(user_interests)
        extended_prompt = f"{prompt}\n### User Interests:\n{user_interest_text}"
        inputs = tokenizer(extended_prompt, return_tensors="pt").to(device)
        logging.info(f"Tokenized input: {inputs}")
        outputs = model.generate(inputs, max_length=300, temperature=0.5, num_return_sequences=1, do_sample=True)
        logging.info(f"Generated outputs: {outputs}")
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logging.info(f"Decoded response: {response}")
        return response
    except Exception as e:
        logging.error(f"Error during generation: {e}")
        return "Error during generation"

# /start
def start(update: Update, context: CallbackContext):
    update.message.reply_text("سلام من مگس هستم")

# private messages
def handle_message(update: Update, context: CallbackContext):
    user_id = str(update.effective_user.id)
    message = update.message.text
    collect_user_info(user_id, message)
    user = user_data[user_id]
    res = search(message, 4, corpus_embeddings, paper_text)
    print(f"Found relevant abstracts: {res}")
    user_interests = user["interests"]
    prompt = f'''
    با توجه به شرایط زیر به این سوال پاسخ دهید:
    {message},
    متن نوشته: {res.iloc[0]['res']} - {res.iloc[1]['res']} - {res.iloc[2]['res']} - {res.iloc[3]['res']}
    '''
    prompt = f"### Human:{prompt}\n### Assistant:"
    response = generate_response(prompt, user_interests)
    print(f"Generated response: {response}")
    update.message.reply_text(response)

# group messages
def handle_group_message(update: Update, context: CallbackContext):
    if should_respond_to_group_message(update.message.text):
        handle_message(update, context)

# should respond in a group or not
def should_respond_to_group_message(message):
    keywords = ['کلمه یک', 'کلمه دو', 'کلمه سه']
    return any(keyword in message for keyword in keywords)

# the bot
TOKEN = "TOKEN"

updater = Updater(TOKEN, use_context=True)
dispatcher = updater.dispatcher
dispatcher.add_handler(CommandHandler("start", start))
dispatcher.add_handler(MessageHandler(Filters.private & Filters.text, handle_message))
dispatcher.add_handler(MessageHandler(Filters.group & Filters.text, handle_group_message))
updater.start_polling()
updater.idle()