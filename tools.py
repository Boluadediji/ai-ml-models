from langchain_classic.agents import Tool
from utils import classify_intent, check_balance, report_card_issues, unsupported

classify_tool = Tool(
    name="ClassifyIntent",
    func=classify_intent,
    description="Classifies user intent in a banking conversation"
)

balance_tool = Tool(
    name="CheckBalance",
    func=check_balance,
    description="Returns account balance for a given account_id"
)

card_tool = Tool(
    name="Report_Card_Issues",
    func=report_card_issues,
    description="Block a card and return next steps"
)

unsupported_tool = Tool(
    name="unsupported",
    func=unsupported,
    description="Returns a message indicating the requested operation is unsupported"
)

tools = [classify_tool, balance_tool, card_tool, unsupported_tool]