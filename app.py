from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_community.llms import OpenAI
from langchain.prompts import ChatPromptTemplate
import os
import re
import json

app = Flask(__name__)
CORS(app)

# Load the OpenAI API Key from environment
api_key = os.getenv("OPENAI_API_KEY")

@app.route('/processPrompt', methods=['POST'])
def process_prompt():
    data = request.get_json()

    llm = OpenAI(api_key=api_key, model="gpt-4o-mini") 
    actual_prompt = data.get('prompt')
    print(f"Received Prompt: {actual_prompt}")

    # Define a prompt template
    extract_fields = ChatPromptTemplate.from_messages([
    ("system", "You are a good assistant, help me in extracting some data from the provided prompt."),
    ("human", """I will provide you a prompt like 'Assign a high-priority AC maintenance task to Amit Patel. 
    The site is a customer named Rajesh Sharma. Schedule it for tomorrow at 11 AM. Client’s phone number is 9876543210' 
    then you have to give me the response in JSON format like:
    {{
    "name": "Amit Patel",
    "phoneNumber": "9876543210",
    "location": "Sector 21, Gurgaon",
    "task": "AC Maintenance",
    "priority": "High",
    "customerName": "Rajesh Sharma"
    }}
    """),
    ("human", "\n\nDo not generate the output based on the above sample — it’s just for reference on format.\n"
              "Now the actual prompt is '{prompt}'. If any field data is missing, keep it blank.\n"
              "If no actual prompt is provided, keep the structure same with empty values, like 'name': '', etc.")
    ])


    # Format the prompt
    messages = extract_fields.format_messages(prompt=actual_prompt)
    final_prompt = "\n\n".join([f"{m.type.upper()}: {m.content}" for m in messages])

    # Call LLM
    result = llm.invoke(final_prompt)
    print(f"Raw LLM Response: {result}")

    # Extract JSON from string using regex
    match = re.search(r'\{.*?\}', result, re.DOTALL)
    if match:
        json_part = match.group()
        result_json = json.loads(json_part)
    else:
        result_json = {
            "error": "Model did not return JSON. Prompt may be incomplete or invalid.",
            "rawResponse": result
        }

    return jsonify(result_json)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)