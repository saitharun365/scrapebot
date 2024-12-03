from dotenv import load_dotenv
import regex as re
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models  import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import HumanMessage, AIMessage
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import nest_asyncio
import uvicorn
import aiofiles
import json
import os

# Load the .env file
load_dotenv()
#OPENAI_API_KEY = os.getenv("openai_api_key")
#ngrok_auth = os.getenv("ngrok_auth")


# Initialize FastAPI app
app = FastAPI()

# Define persistence directories for Chroma
persist_directory_retriever = "chroma_store"
persist_directory_research = "chroma_store1"

# Initialize OpenAI embeddings and Chroma vector stores
embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")
chroma_store = Chroma(
    collection_name="retriever_bot",
    embedding_function=embedding_model,
    persist_directory=persist_directory_retriever
)
chroma_store1 = Chroma(
    collection_name="research_info",
    embedding_function=embedding_model,
    persist_directory=persist_directory_research
)

# Initialize memory
memory = ConversationBufferMemory(memory_key="chat_history",output_key="answer",return_messages=True)

# Global variables
last_intent = None
interaction_count = 0
feedback_requested = False
pending_conversation_data = None  # Variable to hold conversation data awaiting feedback
feedback_timeout_counter = 0       # Counter for feedback prompt retries
MAX_FEEDBACK_PROMPTS = 2           # Maximum number of feedback prompts before giving up


# Define the lists of affirmative and negative responses
affirmative_responses = [
    "yes", "yup", "yeah", "absolutely", "definitely", "sure",
    "yes, this is helpful", "it's helpful", "very helpful", "indeed",
    "positive", "certainly", "of course", "thank you", "this is great", "glad"
]

negative_responses = [
    "no", "nope", "not really", "not at all", "negative",
    "unfortunately not", "it's not helpful", "no, not helpful",
    "no thanks", "nah", "not quite", "don't think so", "disappointed", "bad"
]

async def save_conversation_history(conversation_data):
    # Process the chat_history to make it JSON serializable
    processed_chat_history = []
    for message in conversation_data['chat_history']:
        if isinstance(message, HumanMessage):
            processed_chat_history.append({
                'type': 'human',
                'content': message.content
            })
        elif isinstance(message, AIMessage):
            processed_chat_history.append({
                'type': 'ai',
                'content': message.content
            })
        else:
            # Handle other message types if necessary
            processed_chat_history.append({
                'type': 'other',
                'content': message.content
            })
    # Replace the chat_history with the processed version
    conversation_data['chat_history'] = processed_chat_history

    # Now it's safe to serialize conversation_data to JSON
    async with aiofiles.open("conversation_history.json", mode="a") as file:
        await file.write(json.dumps(conversation_data) + "\n")
def get_last_updated_date(log_file_path):
    try:
        with open(log_file_path, 'r') as file:
            lines = file.readlines()
        for line in reversed(lines):
            match = re.search(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', line)
            if match:
                return match.group(0)
        return "Unknown"
    except Exception as e:
        print(f"Error reading log file: {e}")
        return "Unknown"


def get_parent_intent(intent_name):
    """
    Returns the parent intent for a given intent. If the intent is a parent intent, returns itself.

    Args:
        intent_name (str): The name of the intent to check.

    Returns:
        str: The parent intent name.
    """
    # Intent mapping dictionary
    intent_mapping = {
        "Get_Course_info": {
            "sub_intents": [
                "Get_Course_info - custom"
            ]
        },
        "Get_CPT_OPT_Info": {
            "sub_intents": [
                "Get_CPT_Application_Process",
                "Get_CPT_OPT_Documents",
                "Get_OPT_Application_Process",
                "Get_CPT_Eligibility",
                "Get_OPT_Eligibility"
            ]
        },
        "Get_General_Info": {
            "sub_intents": [
                "Get_General_Info - custom"
            ]
        },
        "Get_Research_Info": {
            "sub_intents": [
                "Get_Research_Faculty_Info",
                "Get_Research_Info - custom"
            ]
        }
    }

    # Iterate through the mapping to find the parent intent
    for parent_intent, details in intent_mapping.items():
        if intent_name == parent_intent or intent_name in details["sub_intents"]:
            return parent_intent

    # If no match is found, return None
    return intent_name

def handle_intent_change(intent_name: str):
    """
    Handles changes in user intent and manages interaction counts.
    Sets feedback_requested flag when intent changes or interaction count exceeds threshold.
    """
    global last_intent, interaction_count, feedback_requested, pending_conversation_data, feedback_timeout_counter

    if last_intent is not None and last_intent != intent_name:
        # Intent has changed
        feedback_requested = True
        feedback_timeout_counter = 0  # Reset the feedback timeout counter
        # Store the previous conversation data
        pending_conversation_data = {
            "intent": last_intent,
            "chat_history": memory.chat_memory.messages.copy(),
            "feedback": None
        }
        interaction_count = 0  # Reset interaction count for new intent
    else:
        # Same intent, increment interaction count
        interaction_count += 1
        if interaction_count >= 5:
            feedback_requested = True
            feedback_timeout_counter = 0  # Reset the feedback timeout counter
            # Store the current conversation data
            pending_conversation_data = {
                "intent": intent_name,
                "chat_history": memory.chat_memory.messages.copy(),
                "feedback": None
            }
            interaction_count = 0  # Reset after requesting feedback

    last_intent = intent_name  # Update to the new intent

def user_is_asking_for_link(query):
    keywords = ["link", "source", "reference","find" , "references","sources","url", "where can I find", "can you provide the link"]
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in keywords)

def extract_links_from_source_docs(source_docs):
    links = []
    seen_links = set()  # Set to track unique links

    for doc in source_docs:
        try:
            metadata = doc.metadata if isinstance(doc.metadata, dict) else {}
            link = metadata.get("link")
            title = metadata.get("title")

            # Only add unique links
            if link and link not in seen_links:
                links.append((title, link))
                seen_links.add(link)
        except AttributeError:
            # Handle the case where doc.metadata is not accessible
            continue

    return links
# Load LLM model
def load_llm():
    return ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)

def get_user_messages(memory):
    user_messages = []
    for msg in memory.chat_memory.messages:
        if msg.type == 'human':
            user_messages.append(msg.content)
    return '\n'.join(user_messages)

def load_prompt(intent_name):
    if intent_name == "Get_Course_info":
        prompt = """You need to answer the user's question about course related and  prerequisites information.
         Use the conversation history to provide a consistent response.
        Conversation History: {chat_history}
        Context: {context}
        Question: {question}"""
    elif intent_name == "Get_CPT_OPT_info":
        prompt = """You need to answer the user's question about CPT or OPT information based on the given context only.
        Give friendly responses.
        Use the conversation history to provide a consistent response.
        Conversation History: {chat_history}
        Context: {context}
        Question: {question}"""
    elif intent_name == "Get_General_info":
        prompt = """You are a helpful assistant for students, here to provide accurate answers strictly based on the provided context.
        Do not create or infer any information that isn’t explicitly in the context. If the answer cannot be found within the context, respond with a polite message indicating that additional information is needed and prompt the student to either rephrase their question or clarify. If applicable, you may suggest specific topics or terms that could help refine the search.
        Context: {context}
        Student's Question: {question}
        Use the conversation history to provide a consistent response.
        Conversation History: {chat_history}
        Example Response (when information is missing):
        I couldn't find a direct answer in the provided information. Could you provide more details or rephrase your question? You might also consider specifying terms or topics to help refine the search."""

    elif intent_name == "Get_Research_info":
        prompt = """You need to answer the user's question about research information. Please dont assume answer only based on the context given
        Use the conversation history to provide a consistent response.
        Conversation History: {chat_history}
        Context: {context}
        Question: {question}"""
    else:
        prompt = """You need to answer the user's question.
        Use the conversation history to provide a consistent response.
        Conversation History: {chat_history}
        Context: {context}
        Question: {question}"""
    return ChatPromptTemplate.from_template(prompt)

def is_syllabus_query(query: str) -> bool:
    """
    Determines if the query is related to syllabus information.

    :param query: User's query string.
    :return: True if syllabus-related, False otherwise.
    """
    syllabus_keywords = ["syllabus", "course outline", "course structure", "course syllabus", "course outline", "course outcomes"]
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in syllabus_keywords)


def get_full_course_title(course_input):
    course_mapping = {
  "data 601": "DATA 601 Introduction to Data Science",
  "data 602": "DATA 602 Introduction to Data Analysis and Machine Learning",
  "data 603": "DATA 603 Platforms for Big Data Processing",
  "data 604": "DATA 604 Data Management",
  "data 605": "DATA 605 Ethical Legal Issues in Data Science",
  "data 606": "DATA 606 Capstone in Data Science",
  "data 607": "DATA 607 Leadership in Data Science",
  "data 608": "DATA608 Probability and Statistics for Data Science",
  "data 690": "DATA 690 Special Topics: Statistical Analysis and Visualization with Python",
  "data 690": "DATA 690 Special Topics: Mathematical Foundations for Machine Learning",
  "data 690": "DATA 690 Special Topics: Data Structures and Algorithms in Python",
  "data 690": "DATA 690 Special Topics: Applied Machine Learning with MATLAB",
  "data 690": "DATA 690 Special Topics: Designing Data Driven Web Applications",
  "data 690": "DATA 690 Financial Data Science",
  "data 690": "DATA 690 Special Topic: Modern Practical Deep Learning",
  "data 690": "DATA 690 Special Topics: Introduction to Natural Language Processing",
  "data 690": "DATA 690 Special Topics: Artificial Intelligence for Practitioners",
  "data 696": "DATA 696 Independent Study for Interns and Co-op Students",
  "data 699": "DATA 699 Independent Study in Data Science",
  "data 613": "DATA 613  Data Visualization and Communication",
  "data 621": "DATA 621  Practical Deep Learning"}
    # Normalize input
    course_input_lower = course_input.lower()

    # First, try exact match in keys (course IDs)
    for course_id in course_mapping.keys():
        if course_id.lower() == course_input_lower:
            return course_mapping[course_id]

    # Then, try exact match in values (course titles)
    for course_title in course_mapping.values():
        if course_title.lower() == course_input_lower:
            return course_title

    # No match found
    return None

async def retrieve_syllabus_documents(course_titles):
    # Build the metadata filter
    metadata_filter = {"title": {"$in": course_titles}}

    # Perform similarity search with the filter
    try:
        results = chroma_store.similarity_search("", k=5, filter=metadata_filter)
        return results
    except Exception as e:
        print(f"Error during syllabus retrieval: {e}")
        return []
# Load LLM model
def load_llm():
    return ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)

async def handle_syllabus_query(query: str, parameters: dict,memory) -> dict:
    # Extract course name from parameters
    course_inputs = parameters.get("Course_name", [])
    if not course_inputs:
        return {"answer":"Please specify the course name or ID for which you want the syllabus."}
    print(course_inputs)
    # Get the full course title from the input
    course_titles = [get_full_course_title(course_input) for course_input in course_inputs]
    print('Here are the course titles: ',course_titles)
    # Retrieve syllabus documents
    retrieved_docs = await retrieve_syllabus_documents(course_titles)
    if not retrieved_docs:
        return {"answer": "Sorry, I couldn't find the syllabus for the specified course."}

    # Prepare the context from the retrieved documents
    context = "\n".join([doc.page_content for doc in retrieved_docs])
   # Construct the prompt for the LLM
    prompt = f"""
    You are a helpful assistant that organizes syllabus information into a clear and concise tabular format. Based on the context provided, generate a week-by-week syllabus for each course mentioned in the query. Use readable and well-structured text in the table.

        Context:{context}
        Question: {query}
        Format the answer as follows:
        | Week | Topic Description |
        |------|-------------------|
        | 1    | Topic details     |
        | 2    | Topic details     |
        ...
        Ensure that:
        1. The information is concise and well-organized.
        2. If a course does not have sufficient details in the context, indicate it with "Details not available."
        Answer:
        """
    # Generate the answer using the LLM
    try:
        response =  load_llm().invoke([prompt])
        memory.chat_memory.add_user_message(query)
        memory.chat_memory.add_ai_message(response)
    except Exception as e:
        print(f"Error during LLM generation: {e}")
        return {"response": "An error occurred while generating the syllabus response."}

    return {"answer": response.content, "source_documents": retrieved_docs}

# Function to select the retriever based on the intent
def select_retriever(intent_name):
    if intent_name == "Get_Research_info":
        return chroma_store1.as_retriever(search_type="similarity", search_kwargs={'k': 5})
    else:
        return chroma_store.as_retriever(search_type="similarity", search_kwargs={'k': 5})

# Webhook route for DialogFlow fulfillment
@app.post("/webhook")
async def webhook(request: Request):
    global interaction_count, feedback_requested, pending_conversation_data, last_intent, feedback_timeout_counter
    # Extract the query and intent from DialogFlow request
    req = await request.json()
    query = req.get("queryResult", {}).get("queryText", "")
    intent_name = req.get("queryResult", {}).get("intent", {}).get("displayName", "")
    parameters = req.get("queryResult", {}).get("parameters", {})
    # Handle the Default Welcome Intent
    if intent_name == "Default Welcome Intent":
        memory.chat_memory.clear()
        log_file_path = "C:/Users/Surya/scraping_log.log"
        last_updated_date = get_last_updated_date(log_file_path)
        response_text = (
                f"Hello! I was last updated on {last_updated_date}. "
                "While I strive to provide accurate and up-to-date details, there may have been changes since this date. How may I assist you today?"
            )
        return JSONResponse(content={"fulfillmentText": response_text})
    intent_name = get_parent_intent(intent_name)
    print(f"Received query: {query} | Intent: {intent_name}")

    # Check if the user has provided feedback
    # **Step 1: Handle Feedback If Previously Requested**
    if feedback_requested and pending_conversation_data:
        user_input = query.lower().strip().replace("'", "").replace("\"", "")
        feedback = None
        if user_input in affirmative_responses:
            feedback = "positive"
        elif user_input in negative_responses:
            feedback = "negative"
        else:
            # User has not provided feedback yet
            # Proceed to handle the user's query but keep prompting for feedback
            pass  # We'll handle this below

        if feedback is not None:
            # Update the pending conversation data with feedback
            pending_conversation_data["feedback"] = feedback
            # Save the conversation history with feedback
            await save_conversation_history(pending_conversation_data)
            # Reset flags and clear memory
            feedback_requested = False
            pending_conversation_data = None
            feedback_timeout_counter = 0  # Reset the feedback timeout counter
            memory.chat_memory.clear()
            response_text = "Thank you for your feedback!"
            print("Feedback received and conversation history saved.")
            return {"fulfillmentText": response_text}
        else:
            # User has not provided valid feedback yet
            feedback_timeout_counter += 1
            if feedback_timeout_counter >= MAX_FEEDBACK_PROMPTS:
                # Give up on prompting for feedback after max attempts
                # Save the conversation history without feedback
                await save_conversation_history(pending_conversation_data)
                feedback_requested = False
                pending_conversation_data = None
                feedback_timeout_counter = 0
                memory.chat_memory.clear()
                print("Feedback not provided after maximum attempts. Proceeding without feedback.")
                # Proceed to handle the current query as normal
            else:
                pass

    # **Step 2: Handle Intent Change and Interaction Count**
    handle_intent_change(intent_name)
    # Determine if the query is a syllabus query
    if intent_name == "Get_Course_info" and is_syllabus_query(query):
        # Handle syllabus query separately
        #print("True matches criteria")
        result = await handle_syllabus_query(query, parameters,memory)
        response_text = result['answer']
    else:
      # Select the appropriate retriever and prompt based on intent
      retriever = select_retriever(intent_name)
      prompt = load_prompt(intent_name)

      # Initialize the question generation and document response chains
      qa_chain = ConversationalRetrievalChain.from_llm(
              llm= load_llm(),
              retriever=retriever,
              memory=memory,
              combine_docs_chain_kwargs={'prompt': prompt},
              return_source_documents=True,
              verbose=True,
              rephrase_question=True,
         )
      # Run the query through the QA chain and get the response
      result = qa_chain({"question": query, "chat_history": get_user_messages(memory)})
      response_text = result['answer']

    # Extract source documents
    source_docs = result.get('source_documents', [])

    # Check if the user is asking for the link
    if user_is_asking_for_link(query):
      # Extract links from source documents
      links = extract_links_from_source_docs(source_docs)  # Returns list of URLs or list of (title, URL) tuples
      if links:
        # Append links to the response using Markdown
        response_text += "\n\nHere are the links to the sources:\n\n"

        for link in links:
            if isinstance(link, tuple):
                title, url = link
                response_text += f"- [{title}]({url})\n"
            else:
                response_text += f"- [{link}]({link})\n"
      else:
        response_text += "\n\nSorry, I couldn't find any links to provide."


    # Append the feedback prompt if feedback is requested
    if feedback_requested:
        response_text += "\n\nWas this conversation helpful? Please reply with 'Yes' or 'No'."


    # Prepare DialogFlow response
    response = {"fulfillmentText": response_text}

    return response

# Run the FastAPI app using nest_asyncio
nest_asyncio.apply()
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
