# ReToolSQL: Tool-Calling Reinforcement Learning for Robust Text-to-SQL

## 1. Project Description

**Problem:** NL2SQL agents must answer complex business questions reliably across diverse schemas, but frontier models still struggle with tool use, execution feedback, and robustness. **Context:** At JPMC, NL2SQL is a high-value capability for LOBs including AWM, CIB, and CCB. **Commitment:** This session covers how ReToolSQL combines tool-calling reinforcement learning with inference-time improvements to make Text-to-SQL agents more accurate, reliable, and production-ready.

## 2. Project Benefits

ReToolSQL improves NL2SQL accuracy by training models to use tools more effectively and by structuring inference-time tool calls for better reasoning. Current results beat the BIRD benchmark leaderboard and rank ahead of frontier models such as Opus 4.6 and GPT 5.5. For JPMC, this translates into higher-confidence analytics, fewer manual SQL handoffs, faster business question answering, and a stronger foundation for production NL2SQL agents across multiple LOBs.

## 3. Required Skill Set

Attendees should have a general understanding of generative AI, data analytics, and how SQL is used to answer business questions. Familiarity with machine learning concepts, LLM agents, or production AI systems is helpful but not required. The session is designed for engineers, data scientists, product owners, and technology leaders who want to understand how reinforcement learning and tool use can improve enterprise AI applications.

## 4. Who Is Using This Solution

The solution benefits teams building or consuming NL2SQL capabilities across JPMC. We currently have a production NL2SQL agent used by multiple LOBs, including AWM, CIB, and CCB. Business users benefit from faster access to data insights, engineering teams benefit from more reliable agent behavior, and data platform teams benefit from a reusable approach for improving SQL generation across schemas, tools, and workflows.

## 5. Call to Action

Attendees will leave with a practical way to think about improving enterprise LLM agents beyond prompting alone. They will be able to identify where tool-calling reinforcement learning and inference-time design can make agentic systems more reliable, measurable, and useful for business-critical workflows such as NL2SQL.

## 6. Why Now for DevUp?

Enterprise AI is moving from demos to production systems, and NL2SQL is one of the clearest places where reliability matters. JPMC already has production usage across multiple LOBs, while benchmark results show that targeted RL and smarter tool calls can outperform general frontier models. DevUp is the right moment to share how practical training and inference choices can close the gap between impressive models and dependable enterprise agents.

## 7. Key Takeaways

- **Understand** how tool-calling reinforcement learning improves Text-to-SQL robustness.
- **Apply** inference-time tool design patterns to make agent outputs more reliable.
- **Evaluate** NL2SQL systems using benchmark performance and production-readiness criteria.
- **Recognize** where RL can outperform prompt-only optimization for enterprise agents.
- **Translate** research results into reusable patterns for business-facing AI systems.

## 8. Session Outline

- Introduce the ReToolSQL approach and system framing.
- Walk through the tool-calling RL training setup.
- Explain inference-time tool design and test-time improvements.
- Share benchmark results on BIRD and planned Spider evaluation.
- Connect benchmark gains to production NL2SQL usage at JPMC.
- Discuss lessons learned from scaling agent reliability.
- Close with practical patterns for enterprise AI teams.



Hi, I’m Pratik, one of the lead engineers from the FineTuning team at CDAO. I’ll be doing this talk with my colleague, Chandra Dhir.

Our talk is about ReToolSQL: Tool-Calling Reinforcement Learning for Robust Text-to-SQL.

Natural Language to SQL is one of the most important AI use cases across the firm. Many teams want business users, analysts, and agents to ask questions in plain English and get accurate answers from enterprise data. But in practice, Text-to-SQL is hard. Schemas are complex, business logic is subtle, and even frontier models can fail when they need to reason, inspect data, or recover from execution errors.

That is where tool-calling reinforcement learning comes in.

In this work, we train models not just to generate SQL, but to use tools more intelligently at inference time: checking queries, inspecting schema details, repairing mistakes, and improving reliability before producing the final answer.

The results are exciting. We are beating the BIRD benchmark leaderboard and ranking ahead of frontier models offered from companies like Google, Anthropic, OpenAI, Databricks, and Snowflake.

Why trust this talk? I have experience fine-tuning models of different sizes, from 2B up to 32B, across model families including Ministral, GPT-OSS, and Gemma. This work brings together practical fine-tuning, RL training, agent design, and test-time tool use.

If you are building agents, analytics copilots, data assistants, or workflows where models need to call tools and make reliable decisions, this session is for you. You will learn how to identify use cases where RL or tool-calling RL can improve agentic workflows, when smarter inference-time design matters, and how smaller trained models can sometimes reduce token usage, latency, and cost while competing with much larger frontier systems.

The goal is simple: move beyond prompting alone and learn how to make enterprise agents more accurate, efficient, and production-ready.

See you at DevUp.
