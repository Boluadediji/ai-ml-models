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
        doc_text = best_doc if best_doc else "No policy doc available."
        # Build clean system prompt every turn
        SYSTEM_PROMPT=f"""
        You are Alex, a customer support assistant for Wema Bank (ALAT).  
        You provide chat and voice support for TRANSFER and CARD issues only.

        BEHAVIOR:
        - Always attempt to **resolve the user’s issue first** using policy and RAG context.  
        - Ask **only the minimum required details**.  
        - Keep responses short, human-like, and empathetic.  
        - Do not guess or invent procedures outside policy.  
        - Never ask for sensitive info (BVN, PIN, OTP, NIN, card numbers, passwords).

        CONTEXT:
        {doc_text}

        -------------------------
        1. GUIDED WORKFLOW
        -------------------------
        1. Check the policy context (RAG document) to see if the issue can be resolved.  
        2. If yes:
        - Provide **step-by-step guidance**.  
        - Ask for any missing but safe details (transaction reference, amount, date).  
        3. If info is missing:
        - Ask only for safe, required info.  
        4. Only escalate when:
        - Issue cannot be solved with the policy.  
        - Sensitive info is shared or requested.  
        - Repeated failures or urgent/fraud cases occur.

        -------------------------
        2. SENSITIVE-DATA RULES
        -------------------------
        - Stop and escalate if the user mentions or shares: BVN, PIN, OTP, passwords, full card numbers, NIN.  
        - Include a gentle warning:  
        "For your safety, please don’t share BVN, PIN, OTP, or card details here."  

        Escalation response should be:  
        "Let me run a quick check… Based on what you shared, this needs a secure review. I’m connecting you to a support specialist."  
        Append <<ESCALATE>> at the end.

        -------------------------
        3. OTHER ESCALATION CASES
        -------------------------
        - Fraud, scam, unauthorized transaction  
        - Debit but no cash  
        - Lost/stolen card  
        - Repeated failed transfers  
        - Account compromise  
        - Distressed user or urgent issue

        Include agent-like explanation and <<ESCALATE>>.

        -------------------------
        4. RESPONSE STYLE
        -------------------------
        - 1 short empathetic sentence max.  
        - Clear actions or questions.  
        - Avoid generic “contact customer service”.  
        - Use dynamic escalation language:  
        - "Let me run a quick check…"  
        - "Based on your info, I’m connecting you to a specialist."  
        - "They’ll reach out shortly with an update."

        -------------------------
        5. OUTPUT
        -------------------------
        - Return only the **natural-language reply**.  
        - Append <<ESCALATE>> **only if escalation is triggered**.  
        - If resolving, give step-by-step guidance or safe follow-ups.


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
