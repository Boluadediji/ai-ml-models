import traceback
from config import get_azure_client
from intent_classifier import classify_intent
from ragengine import retrieve_relevant_doc
import os

client = get_azure_client()


def chatbot():
    print("Hello! I am Alex, your banking assistant. How can I help you? (type 'q' to quit)")

    conversation_history = []     # store only user + assistant messages
    session_intent = None         # classify intent once

    while True:
        user_text = input("\nYou: ").strip()

        if user_text.lower() in ["q", "quit", "exit"]:
            print("Alex: Thank you for banking with us. Goodbye!")
            break

# ---------- INTENT CLASSIFICATION ----------
        try:
            intent = classify_intent(user_text)
            print(f"(Detected intent: {intent})")

            # Reset session intent if user changes topic
            if session_intent != intent:
                print(f"(Switching session intent from {session_intent} → {intent})")
                session_intent = intent

        except Exception:
            print("Alex: Sorry, I couldn't classify your request.")
            session_intent = "unsupported_request"
            continue

        # ---------- RAG DOCUMENT RETRIEVAL ----------
        try:
            best_doc, score = retrieve_relevant_doc(session_intent, user_text)
            print(f"(Retrieved doc with similarity score: {score:.4f})")
        except Exception:
            print("\nAlex: Sorry, I couldn't retrieve relevant information.")
            traceback.print_exc()
            best_doc = None

        # ---------- LLM RESPONSE GENERATION ----------

        # Build clean system prompt every turn
        SYSTEM_PROMPT="""
        You are Alex, a customer support assistant for Wema Bank.

        You assist with TRANSFER and CARD issues only, using the provided policy context.
        Do not guess or invent procedures outside the policy.

        CONTEXT:
        {doc_text}

        GOAL:
        1. Resolve the issue if fully covered by policy.
        2. If not, gather the minimum required details and prepare the case for escalation.

        WORKFLOW:
        - If policy answers the issue → give clear, step-by-step guidance.
        - If information is missing → ask only what is required to proceed.
        - If policy does not cover the issue → escalate clearly.

        WHEN ASKING QUESTIONS:
        Collect details an agent would need, such as:
        - Issue type (transfer or card)
        - Transaction reference (if any)
        - Date & time of issue
        - Amount
        - Error message or symptom
        - Whether troubleshooting steps were tried

        RESPONSE STYLE:
        - 1 short empathetic sentence max
        - Clear actions or questions
        - No unnecessary explanations
        - No out-of-scope help

        ESCALATION:
        If escalation is required, respond with:
        - A brief summary of the issue
        - A list of collected details
        - Clear next-step contact info
        """
        # Build fresh message list
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Add past conversation
        messages.extend(conversation_history)

        # Add new user message
        messages.append({"role": "user", "content": user_text})

        # LLM call
        try:
            response = client.chat.completions.create(
                model="aicso",
                messages=messages,
                temperature=0,
                max_tokens=250
            )
            reply = response.choices[0].message.content
            print(f"\nAlex: {reply}")

            # Save turn
            conversation_history.append({"role": "user", "content": user_text})
            conversation_history.append({"role": "assistant", "content": reply})

        except Exception:
            print("\nAlex: Sorry, I ran into a problem while generating a response.")
            traceback.print_exc()
            continue


if __name__ == "__main__":
    chatbot()
