from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_community.llms import OpenAI
from langchain.prompts import ChatPromptTemplate
import os
import re
import json

app = Flask(__name__)
CORS(app)

# Now you can access the key
api_key = os.getenv("OPENAI_API_KEY")

@app.route('/processPrompt', methods=['POST'])
def process_prompt():
    data = request.get_json()

    llm = OpenAI(api_key=api_key,model="gpt-4o-mini")
    actual_prompt=data.get('prompt')
    print(f"Received Prompt: {actual_prompt}")

    # actual_prompt="Assign a medium-priority plumbing inspection task to Rohan Mehta. The site is a customer named Priya Verma. Schedule it for tomorrow at 3 PM."
    # actual_prompt="give a task for ac repair to Shivam Singh and location is noida sector-32 , shop no.-13. name of the client is Ankit kumar and his phone number is 9876542319, and keep the priority high"
    # actual_prompt="visit ke liye task bnaao garg customer ke vaha ka uska number 9818097594 hai "

    # Define a prompt template
    extract_fields = ChatPromptTemplate.from_messages([
        ("system", "You are a good assistant, help me in extracting some data from prompt that is provided."),
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
    }}""" ),
    ("human","\n\n\n\n do not generate the output based on the above provided sample that is only for reference how to generate response \n\n\nnow the actual prompt is '{prompt}' and if any field data is missing than keep it blank \
     \n\n\n if there is nothing provided in quotes '' after actual prompt keyword than keep the structure of response same and leave the values blank for eg- 'name':'',etc ")
    ])

    # Get the formatted message list (without variables)
    messages = extract_fields.format_messages(prompt=actual_prompt)

    # Concatenate all message contents into a single string
    final_prompt = "\n\n".join([f"{m.type.upper()}: {m.content}" for m in messages])

    result=llm.invoke(final_prompt)
    print(f"Raw LLM Response: {result}")


    # Extract JSON from string using regex
    # json_part = re.search(r'\{.*\}', result, re.DOTALL).group()
    match = re.search(r'\{.*?\}', result, re.DOTALL)
    if match:
        json_part = match.group()
        result_json = json.loads(json_part)
    else:
        result_json = {
    # # Convert string JSON to Python dict
            "error": "Model did not return JSON. Prompt may be incomplete or invalid.",
            "rawResponse": result
        }

    return jsonify(result_json)

if __name__ == '__main__':
    # process_prompt()
    app.run(debug=True)
