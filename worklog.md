# iagent
IAgent is multi-agent library, not a framework. It aims to provide primitives to enable you build agent applications, than locking you in a specific framework.

# project structure
src/iagent
tests/iagent

# Env, Tool, Session, Task and Agent
## Env
Env conceptually provide the scope or environment the agents interact with, particularly through using tools. Tools and Agents (agent can be viewed extending to tools) register their capability in Env. When a tool is used, it acts on Env and Env provides the (maybe partial) observability on tool's effect (response) to who uses the tool. In a sense, one can think Env owning tools, and aware of agents, providing discoveries for agents. Though it doesn't meddle the actual communications among agents, it provides information on what protocol how agent can communicate with each other and use tools.

## Agent
Serveral Agents, one particular could be UserProxyAgent, can work together in a Session on one or manay Tasks. An Agent can have long term memory to improve its future performance on specific tasks or providing personal experience like UserProxyAgent for user.

## Session and Task
Session is for agents working together on one or many tasks. A task can have SubTask etc. A session will have short term memory to faciliate finishing tasks.

# Working task: Implement Tool
Let us first implement concrete case where python function can be a tool. It should have name, description, argument specification. You can refer langchain-core or openai agent sdk implementation
