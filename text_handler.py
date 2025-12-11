import traceback
from config import get_azure_client
from intent_classifier import classify_intent
from ragengine import retrieve_relevant_doc

client = get_azure_client()

MAX_HISTORY = 8


def handle_text_message(
    user_text: str,
    session_state: dict | None = None,
):
    """
    Chatbot handler for HTTP APIs.

    Returns:
        reply (str)
        escalation (bool)
        updated_session_state (dict)
    """

    # -------------------------------
    # Init session state
    # -------------------------------
    if session_state is None:
        session_state = {
            "conversation_history": [],
            "session_intent": None
        }

    conversation_history = session_state["conversation_history"]
    session_intent = session_state["session_intent"]

    # -------------------------------
    # INTENT CLASSIFICATION
    # -------------------------------
    try:
        intent = classify_intent(user_text)
        if session_intent != intent:
            session_intent = intent
    except Exception:
        reply = "Sorry, I couldn't understand your request."
        return reply, False, session_state

    # -------------------------------
    # RAG DOCUMENT RETRIEVAL
    # -------------------------------
    try:
        best_doc, score = retrieve_relevant_doc(session_intent, user_text)
    except Exception:
        traceback.print_exc()
        best_doc = None

    doc_text = best_doc if best_doc else "No policy doc available."

    # -------------------------------
    # SYSTEM PROMPT (with ESCALATE tag)
    # -------------------------------
    SYSTEM_PROMPT = f"""
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

    # -------------------------------
    # BUILD MESSAGES
    # -------------------------------
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(conversation_history)
    messages.append({"role": "user", "content": user_text})

    # -------------------------------
    # LLM CALL
    # -------------------------------
    try:
        response = client.chat.completions.create(
            model="aicso",
            messages=messages,
            temperature=0,
            max_tokens=250
        )
        raw_reply = response.choices[0].message.content.strip()
    except Exception:
        traceback.print_exc()
        return "Sorry, I ran into an issue generating a response.", False, session_state

    # -------------------------------
    # EXTRACT ESCALATION FLAG
    # -------------------------------
    escalation = False
    reply = raw_reply

    if "<<ESCALATE>>" in raw_reply:
        escalation = True
        reply = raw_reply.replace("<<ESCALATE>>", "").strip()

    # -------------------------------
    # UPDATE SESSION
    # -------------------------------
    conversation_history.append({"role": "user", "content": user_text})
    conversation_history.append({"role": "assistant", "content": reply})

    conversation_history = conversation_history[-MAX_HISTORY:]

    updated_session_state = {
        "conversation_history": conversation_history,
        "session_intent": session_intent
    }

    return reply, escalation, updated_session_state
