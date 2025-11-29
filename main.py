from dotenv import load_dotenv
from openai import AzureOpenAI
import os

load_dotenv()

def load_policy(data):
    with open(data, 'r', encoding='utf-8') as f:
        content = f.read()
        
    return content


def load_documents(self):
        """Load and combine all policy documents"""
        documents = []
        
        files = {
            'policy.txt': 'General Banking Policy',
            'faq.txt': 'Frequently Asked Questions', 
            'loan_policy.txt': 'Loan Policy Guidelines'
        }

        for filename, doc_type in files.items():
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    content = f.read()
                    documents.append(f"--- {doc_type} ---\n{content}")
            except FileNotFoundError:
                print(f"Warning: {filename} not found")
        
        return "\n\n".join(documents)
















policy = load_policy('policy.txt')
faq = load_policy('faq.txt')
loan_policy = load_policy('loan_policy.txt')

bot_banker = AzureOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint=os.getenv("azure_endpoint"),
    api_key=os.getenv("API_KEY"),
)

def prompting(bot_banker, faq, policy, loan_policy):

    knowledge_base = f"""
    --- FAQ ---
    {faq}

    --- POLICY ---
    {policy}

    --- LOAN POLICY ---
    {loan_policy}
    """
    messages = [{'role':'system', 'content':'You are a helpful banking officer  who answers questions only using the provided bank policies.'}]

    messages.append({"role": "system", "content": f"Here are the official documents you must use for reference:\n{knowledge_base}"}
)

    while True:

        user_input = input('Type your Question here or q to quit: ')

        if user_input.lower() in ['q', 'quit', 'exit']:
            print('exiting...')
            break;
        
        messages.append({'role':'user', 'content':user_input})


        response = bot_banker.chat.completions.create(
                messages=messages,
                max_tokens=4096,
                model="gpt-4o-mini",
                temperature=0
            )
        
        reply = response.choices[0].message.content

        print(f'Bot: {reply}')

        messages.append({'role':'assistant', 'content':reply})


if __name__ == '__main__':
    prompting(bot_banker, faq, policy, loan_policy)