# zeroresponder

Inspiration

Emergencies can happen to anyone at any time, and the ability to assist someone in distress during a medical emergency is crucial. The "Zero Responder" app addresses this by potentially reducing response time, especially considering the average first responder time in the US is approximately 14 minutes. In those critical minutes before professional help arrives, a "Zero Responder" can make a significant impact.

What it does

The "Zero Responder" web app acts as an agent enhanced chat with a RAG (Retrieval-Augmented Generation) grounded Language Model (LLM). Its purpose is to assist laypersons in handling various medical issues before the arrival of first response, providing a user-friendly interface accessible through the web.

How we built it

For the data, to ensure accuracy, we utilized the new LlamaParse PDF parser to extract information from a document on Emergency Medical Services (EMS) guidelines for the RAG data. The app is hosted on AWS EC2, employing LlamaIndex and LangChain, with StreamLit serving as the web front end. By extracting domain-specific text from the RAG database, we utilize the LLM to format step-by-step instructions for a zero-responder to follow.

The agent ensures follow-up with additional steps and guidance.

Challenges we ran into

One of the challenges we encountered was following the documentation on how to connect LlamaIndex to AstraDB.

Accomplishments that we're proud of

Our team successfully developed a demo-able proof of concept for the "Zero Responder" app.

What we learned

During the project, we gained valuable insights into various functions and features of LlamaIndex, Langchain, and Astradb.

What's next for Zero Responder

Zero Responder" includes integrating more agentic behaviors. This improvement will empower Good Samaritans to act effectively, streamline coordination with multiple first responders, and potentially utilize nearby resources to significantly enhance the chances of a victim's survival. The ultimate objective is to approach a response time close to zero!
