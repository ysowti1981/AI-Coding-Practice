# Tool Orchestration Service

## Problem

Build a service that sits between an LLM-based agent and a registry of hundreds of enterprise tools — things like "create a Jira ticket," "query Salesforce contacts," "send a Slack message," etc. Each tool has a name, a natural language description, and a JSON schema for its parameters.

When the agent receives a user request like _"find all open deals over $50k and notify the sales team on Slack,"_ the service needs to:

1. **Identify which tools are relevant** (semantic matching)
2. **Plan the execution order** (e.g., Salesforce query first, then Slack notification)
3. **Execute with proper error handling and observability**

## Design Question

How would you architect this system at a high level? What are the major components, and how do they interact?

## Phase 1
You're designing a Tool Resolution and Orchestration Service for an agentic platform. Hundreds of enterprise tools are registered. An LLM agent sends a natural language request, and your service needs to find the right tools, plan execution, run them, and return results — all while being observable and secure.
Walk me through the architecture. I want to hear about:

1. What services or components exist
2. How they communicate
3. Where data lives and in what form
4. What happens when things fail
