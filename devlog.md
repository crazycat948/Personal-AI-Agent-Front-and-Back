#5/1/2026
Today I redesigned the conversation context logic for my personal AI agent.

Previously, the agent sometimes answered unrelated questions, such as “Who is LeBron James?”, even though the project is supposed to only answer questions related to Yifan. I suspect part of the issue comes from using gpt-4o-mini, which may struggle to consistently follow a long prompt after multiple turns.

To improve this, I changed the architecture from storing all conversation history to maintaining a cleaner context list that only stores relevant question-answer pairs.

The new flow uses an LLM-based three-way classification before answering:

1. Relevant Question

If the user directly asks about Yifan, such as:

“What music do you like?”

The system runs the normal RAG pipeline, retrieves information from the markdown knowledge base, generates an answer, and stores this Q&A pair in the context list.

2. Follow-up Question

If the user asks a follow-up question, such as:

“Why?”
“What about the second one?”

The system uses the latest relevant Q&A pair from the context list, combines it with the current query, retrieves from the markdown knowledge base again, and generates a contextual answer.

Follow-up questions are not stored in the context list, because they do not introduce a new main topic.

If the context list is empty and the user asks a follow-up-style question, the system treats it as unrelated and asks the user to first ask a question about Yifan.

3. Unrelated Question

If the question is unrelated to Yifan, such as:

“Who is LeBron James?”

The system directly returns a boundary response instead of running RAG or generating a general answer.

This makes the agent more focused and reduces the chance of it drifting away from its intended purpose.

I also made some small frontend improvements during this update.

Overall, this new structure should improve both context handling and off-topic query control. It is not a perfect solution yet, but it gives the system a much cleaner foundation and makes the behavior easier to debug and improve later.
