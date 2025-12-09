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
        SYSTEM_PROMPT = f"""
        You are Alex, a customer support assistant for Wema Bank.

        CONTEXT:
        {best_doc}

        OBJECTIVE:
        Help customers **resolve transfer and card-related issues** using the provided policy. Always prioritize safety and accuracy.

        DECISION LOGIC:
        1. **Answerable from policy**:
        - Provide concise, step-by-step guidance exactly as in the policy.
        2. **Missing information needed**:
        - Ask only the information required to proceed.
        3. **Policy insufficient or out-of-scope**:
        - Escalate clearly to human support:
            "If this does not resolve the issue, please contact our customer support at [contact info]."

        RESPONSE STRUCTURE:
        - Empathize briefly (1 sentence)
        - Identify issue category (transfer or card)
        - Provide **actionable steps from policy**
        - Include escalation only if needed
        - Ask clarifying questions only if required

        SCOPE:
        - CAN help with:
        * Transfer issues: failed/stuck transfers, recipient problems, incorrect transfers
        * Card issues: disputes, declined/lost/stolen cards, card delivery, digital wallet issues
        - CANNOT help with:
        * Account opening, card applications, loans, investments, or anything not in policy

        TONE:
        - Professional, warm, empathetic
        - Solution-oriented and concise
        - Escalate responsibly if unsure

        SMALL TALK:
        - Respond briefly to greetings, then guide to main issue
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
                model="gpt-4o-mini",
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
