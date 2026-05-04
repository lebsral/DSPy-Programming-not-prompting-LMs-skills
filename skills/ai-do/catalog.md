# Extended Skill Catalog

Every skill in the repo with its full description. This is the authoritative routing reference — ai-do uses these descriptions to match user requests to skills, especially when skills are not installed and their SKILL.md cannot be read.

## Problem-first skills (`ai-` prefix)

For users who describe what they need in plain language.

### Building AI features

#### `/ai-kickoff`
Scaffold a new AI feature powered by DSPy. Use when adding AI to your app, starting a new AI project, building an AI-powered feature, setting up a DSPy program from scratch, or bootstrapping an LLM-powered backend. Also used for DSPy quickstart, DSPy hello world, first DSPy program, getting started with DSPy, new to AI development, add AI to existing Python app, AI feature from zero to working, scaffold AI project structure, best practices for AI project setup, where do I even begin with LLMs, AI boilerplate code, starter template for AI features, bootstrap AI backend, simple AI project template, how to structure an AI codebase, AI mvp in a day, proof of concept AI feature, DSPy project structure best practices.

#### `/ai-planning`
Plan a multi-phase AI feature before building it. Use when you have a PRD or project idea and need to figure out the execution order, which skills to use in what sequence, or how to break an ambitious AI project into phases. Also use when you want to scope an AI feature, create a phased rollout plan, or figure out dependencies between AI components., help me figure out how to execute this, plan my AI feature, what order should I build this in, AI project roadmap, break this into phases, scope an AI feature, phased AI rollout, AI feature planning, multi-phase AI project, AI project dependencies, which skills do I need, AI execution plan.

#### `/ai-choosing-architecture`
Pick the right DSPy module and architecture for your AI feature. Use when you are not sure whether to use Predict, ChainOfThought, ReAct, or a pipeline, need to choose between DSPy patterns, want architecture advice for your AI feature, or are deciding between a single module and a multi-step pipeline. Also use for which DSPy module should I use, Predict vs ChainOfThought, when to use ReAct, single module vs pipeline, DSPy architecture decision, CoT vs PoT vs ReAct, do I need a pipeline, module selection guide, DSPy pattern selection, how to structure my DSPy program.

#### `/ai-sorting`
Auto-sort, categorize, or label content using AI. Use when sorting tickets into categories, auto-tagging content, labeling emails, detecting sentiment, routing messages to the right team, triaging support requests, building a spam filter, intent detection, topic classification, or any task where text goes in and a category comes out. Also use when classification accuracy varies between runs or semantically close categories get confused., auto-categorize support tickets, AI labeling system, text classification with LLM, auto-tag content, email routing with AI, intent classification, sentiment analysis with DSPy, spam detection with AI, topic modeling with LLM, build a classifier without training data, zero-shot classification, AI triage system.

#### `/ai-searching-docs`
Build AI that searches your documents and answers questions. Use when you need search over help docs, knowledge base Q&A, RAG, document retrieval, or a chatbot that answers from your own content. Also used for embedding search loses critical context, retrieval returns irrelevant results, the right document is buried at position 15, RAG pipeline, search our help center, chat with our knowledge base, find answers in our documentation, vector search for documents, semantic search over company docs, AI-powered knowledge base, customer support search, internal document search, answer questions from PDFs, search across multiple data sources.

#### `/ai-querying-databases`
Build AI that answers questions about your database. Use when you need text-to-SQL, natural language database queries, a data assistant for non-technical users, AI-powered analytics, plain English database search, or a chatbot that talks to your database. Covers DSPy pipelines for schema understanding, SQL generation, validation, and result interpretation., text-to-SQL that actually works, AI SQL generation is unreliable, let non-technical users query data, build a data analyst chatbot, business intelligence with AI, self-service analytics, AI dashboard queries, ask questions about my database in English, SQL copilot, AI-powered data exploration, Metabase alternative with AI, chat with your Postgres, natural language analytics, data chatbot for stakeholders.

#### `/ai-summarizing`
Condense long content into short summaries using AI. Use when summarizing meeting notes, condensing articles, creating executive briefs, extracting action items, generating TL;DRs, creating digests from long threads, summarizing customer conversations, or turning lengthy documents into bullet points. Also used for AI summary too generic, summarize Slack threads, condense customer feedback, meeting transcript summary, executive summary generator, AI-powered digest, summarize legal documents, TLDR for long emails, abstractive summarization, extractive summary with AI, bullet point summary from long text, summarize research papers, call transcript summary, weekly digest generator, summarize support tickets, AI loses important details when summarizing, key takeaways extraction.

#### `/ai-parsing-data`
Pull structured data from messy text using AI. Use when parsing invoices, extracting fields from emails, scraping entities from articles, converting unstructured text to JSON, extracting contact info, parsing resumes, reading forms, pulling data from transcripts (VTT, LiveKit, Recall), extracting fields from Langfuse traces, or any task where messy text goes in and clean structured data comes out. Also use when emails are messy and lack structure, or structured data extraction from unstructured content is unreliable., extract entities from text, parse PDF with AI, structured extraction from unstructured text, OCR plus AI extraction, convert email to structured data, pull fields from documents automatically, AI data entry automation, invoice parsing, resume parsing with AI, medical record extraction.

#### `/ai-taking-actions`
Build AI that takes actions, calls APIs, and does things autonomously. Use when you need AI to call APIs, use tools, perform calculations, search the web and act on results, interact with databases, or do multi-step tasks. Also AI that does things not just talks, tool-using AI agent, AI calls external APIs, function calling with DSPy, build AI that books appointments, AI workflow automation, agent that searches and acts on results, AI that updates databases, autonomous AI agent, AI performs multi-step tasks, give LLM access to tools, agentic AI workflow, AI agent for DevOps, build AI assistant that takes actions, MCP tool integration with AI, AI that can browse and click, LLM with tool access.

#### `/ai-writing-content`
Generate articles, reports, blog posts, or marketing copy with AI. Use when writing blog posts, creating product descriptions, generating newsletters, drafting reports, producing marketing copy, creating documentation, writing email campaigns, or any task where AI writes long-form content from a topic or brief. Powered by DSPy content generation pipelines., AI blog writer, generate marketing copy with AI, AI content is too generic and bland, product description generator, AI writes like a robot, make AI match our brand voice, newsletter generator, AI copywriting tool, SEO content generation, bulk content creation with AI, AI ghostwriter, press release generator, email campaign content with AI, AI writes boring content, content pipeline at scale, editorial AI assistant, long-form AI content generation.

#### `/ai-reasoning`
Make AI solve hard problems that need planning and multi-step thinking. Use when your AI fails on complex questions, needs to break down problems, requires multi-step logic, needs to plan before acting, gives wrong answers on math or analysis tasks, or when a simple prompt is not enough for the reasoning required. Covers ChainOfThought, ProgramOfThought, MultiChainComparison, and Self-Discovery reasoning patterns in DSPy., AI gives shallow answers, LLM does not think before answering, chain of thought prompting, make AI show its work, AI fails at math, complex analysis with LLM, multi-step problem solving, AI reasoning errors, LLM logic mistakes, think step by step DSPy, AI cannot do basic arithmetic, deep reasoning with language models, self-consistency for better answers, tree of thought.

#### `/ai-building-pipelines`
Chain multiple AI steps into one reliable pipeline. Use when your AI task is too complex for one prompt, you need to break AI logic into stages, combine classification then generation, do multi-step reasoning, build a compound AI system, orchestrate multiple models, or wire AI components together. Also used for LangChain LCEL alternative, how to chain LLM calls together, one prompt is not enough, multi-step AI workflow, AI pipeline that actually works in production, prompt chaining keeps breaking, DAG of LLM calls, extract then classify then generate, compound AI system design, how to combine multiple AI steps without spaghetti code.

#### `/ai-building-chatbots`
Build a conversational AI assistant with memory and state. Use when you need a customer support chatbot, helpdesk bot, onboarding assistant, sales qualification bot, FAQ assistant, or any multi-turn conversational AI. Also used for chatbot remember previous messages, conversational AI keeps forgetting context, build a helpdesk bot that actually works, chatbot drops context after a few turns, Intercom bot alternative, Zendesk AI alternative, build WhatsApp bot, Slack bot with AI, chatbot escalation to human agent, LangChain chatbot but simpler, chatbot for SaaS onboarding flow.

#### `/ai-coordinating-agents`
Build multiple AI agents that work together. Use when you need a supervisor agent that delegates to specialists, agent handoff, parallel research agents, support escalation (L1 to L2), content pipeline (writer + editor + fact-checker), or any multi-agent system. Also used for CrewAI alternative, AutoGen alternative, LangGraph multi-agent, agents that talk to each other, specialist agents with a supervisor, agents keep stepping on each other, build an AI team, route tasks to the right agent, when one agent is not enough, parallel agents for research.

#### `/ai-scoring`
Score, grade, or evaluate things using AI against a rubric. Use when grading essays, scoring code reviews, rating candidate responses, auditing support quality, evaluating compliance, building a quality rubric, running QA checks against criteria, assessing performance, rating content quality, or any task where you need numeric scores with justifications. Also use when building an LLM as a judge, automated grading system, AI rubric scoring, code review scoring automation, quality assessment automation, compliance scoring, NPS analysis with AI, performance review scoring, score and rank with explanations, build a rating system with AI, automated QA scoring, or judge AI outputs programmatically.

#### `/ai-decomposing-tasks`
Break a failing complex AI task into reliable subtasks. Use when your AI works on simple inputs but fails on complex ones, extraction misses items in long documents, accuracy degrades as input grows, AI conflates multiple things at once, results are inconsistent across input types, you need to chunk long text for processing, or you want to split one unreliable AI step into multiple reliable ones. Also used for one prompt trying to do too much, AI accuracy drops on long inputs, chunking strategy for LLM, divide and conquer for AI, AI cannot handle complex documents, break down AI task into steps, extraction misses items in long text, prompt does too many things at once, map-reduce pattern for LLM, how to split AI work into subtasks, AI overwhelmed by long context, multi-step extraction pipeline.

#### `/ai-moderating-content`
Auto-moderate what users post on your platform. Use when you need content moderation, flag harmful comments, detect spam, filter hate speech, catch NSFW content, block harassment, moderate user-generated content, review community posts, filter marketplace listings, or route bad content to human reviewers. Also used for build content moderation system, UGC moderation at scale, user-generated content filter, trust and safety tooling, hate speech detection model, NSFW detection API, toxic comment classifier, automated abuse detection, report and flag system with AI, content policy enforcement, marketplace listing moderation, DSPy classification with severity scoring, confidence-based routing, reward-based policy enforcement.

### Quality and reliability

#### `/ai-improving-accuracy`
Measure and improve how well your AI works. Use when AI gives wrong answers, accuracy is bad, responses are unreliable, you need to test AI quality, evaluate your AI, write metrics, benchmark performance, optimize prompts, improve results, or systematically make your AI better. Also used for spent hours tweaking prompts, trial and error prompt engineering is not working, quality plateaued early, stale prompts everywhere in your codebase, my AI is only 60% accurate, how to measure AI quality, AI evaluation framework, benchmark my LLM, prompt optimization not working, systematic way to improve AI, AI accuracy plateaued, DSPy optimizer tutorial, MIPROv2 optimization, how to go from 70% to 90% accuracy.

#### `/ai-auditing-code`
Review DSPy code for correctness and best practices. Use when you want a code review of your DSPy program, need to check if your AI code follows best practices, want to find anti-patterns in your DSPy usage, or need a quality audit of your AI implementation. Also use for DSPy code review, is my DSPy code correct, review my AI code, best practices check, DSPy anti-patterns, code quality audit, am I using DSPy right, sanity check my AI code, peer review my DSPy program, does this follow DSPy conventions.

#### `/ai-making-consistent`
Make your AI give the same answer every time. Use when AI gives different answers to the same question, outputs are unpredictable, responses vary between runs, you need deterministic AI behavior, or your AI is unreliable. Also used for same input gives different output every time, prompt sensitivity causes output changes with minor wording tweaks, reordering examples shifts accuracy dramatically, same prompt gives different results every run, AI is non-deterministic, need reproducible AI results, LLM output keeps changing, how to make LLM deterministic, consistent JSON from LLM, reduce output variance, AI flaky in production, stable AI outputs for production.

#### `/ai-checking-outputs`
Verify and validate AI output before it reaches users. Use when you need guardrails, output validation, safety checks, content filtering, fact-checking AI responses, catching hallucinations, preventing bad outputs, or quality gates. Also used for - AI output looks right but is wrong, how to validate JSON from LLM, LLM returns invalid data, catch bad AI outputs before users see them, output quality gate, AI guardrails for production, verify LLM did not hallucinate fields, post-processing LLM responses. Uses dspy.Refine (iterative with feedback) and dspy.BestOfN (sampling, pick best).

#### `/ai-stopping-hallucinations`
Stop your AI from making things up. Use when AI invents information, fabricates facts, is not grounded in real data, needs citations, does not cite sources, generates factually incorrect responses, or gives confident but wrong answers. Also used for LLM generates responses that are factually incorrect or disconnected from the input, how do I ground responses in source docs, AI makes stuff up, hallucination detection, factual grounding, citation generation, source attribution, prevent fabricated facts, verify AI claims against data, RAG grounding, make AI only answer from provided context, AI confidently wrong.

#### `/ai-following-rules`
Make your AI follow rules and policies. Use when your AI breaks format rules, violates content policies, ignores business constraints, outputs invalid JSON, exceeds length limits, includes forbidden content, or does not comply with your specifications. Also use when LLM JSON output is unreliable, you get inconsistent formatting with random spaces and line breaks, or there is extraneous text and conversational fluff around the JSON. Covers dspy.Refine and dspy.BestOfN for hard and soft rule enforcement, content policies, format enforcement, retry mechanics, and composing multiple constraints. Also used for - AI will not follow my system prompt, LLM keeps breaking format, enforce JSON schema on AI output, AI generates prohibited content, constraint violation from LLM, make AI obey business rules, AI ignores my constraints.

#### `/ai-generating-data`
Generate synthetic training data when you do not have enough real examples. Use when you are starting from scratch with no data, need a proof of concept fast, have too few examples for optimization, cannot use real customer data for privacy or compliance, need to fill gaps in edge cases, have unbalanced categories, added new categories, or changed your schema. Also used for create training data with AI, not enough examples to train, augment small dataset, generate labeled examples from scratch, cold start problem for AI, need data but cannot label manually, privacy-safe synthetic data, test data generation for ML, create diverse training examples, data augmentation for NLP, bootstrap dataset from nothing, DSPy synthetic data generation, quality filtering, bootstrapping from zero.

#### `/ai-fine-tuning`
Fine-tune models on your data to maximize quality and cut costs. Use when prompt optimization hit a ceiling, you need domain specialization, you want cheaper models to match expensive ones, you heard fine-tuning will make us AI-native, you have 500+ training examples, or you need to train on proprietary data. Also use when you have spent weeks of manual iteration with no systematic improvement path, or manual prompt tuning got you to a working system but quality plateaued. Covers DSPy BootstrapFinetune, BetterTogether, model distillation, and when to fine-tune vs optimize prompts, LoRA vs full fine-tune, when to fine-tune vs few-shot, distill GPT-4 into a smaller model, teacher-student model training, custom model training with DSPy, model distillation, make a cheap model as good as GPT-4.

#### `/ai-testing-safety`
Find every way users can break your AI before they do. Use when you need to red-team your AI, test for jailbreaks, find prompt injection vulnerabilities, run adversarial testing, do a safety audit before launch, prove your AI is safe for compliance, stress-test guardrails, or verify your AI holds up against adversarial users. Covers automated attack generation, iterative red-teaming with DSPy, and MIPROv2-optimized adversarial testing., red team my AI before launch, find AI vulnerabilities, adversarial testing for LLM, prompt injection attacks, jailbreak testing, AI safety compliance, SOC2 AI audit, OWASP LLM top 10, penetration testing for AI, stress test AI guardrails, can users break my AI, AI safety for regulated industries, test AI before shipping, adversarial prompt dataset.

### Production and operations

#### `/ai-serving-apis`
Put your AI behind an API. Use when you need to deploy your AI as a service, wrap DSPy in FastAPI, serve to frontend, build an AI endpoint, or productionize an optimized DSPy program. Also used for deploy AI as endpoint, FastAPI with DSPy, serve AI to frontend, productionize DSPy, AI microservice, REST API for AI, deploy optimized DSPy program, AI backend service, serve predictions via HTTP, containerize AI service.

#### `/ai-cutting-costs`
Reduce your AI API bill. Use when AI costs are too high, API calls are too expensive, you want to use cheaper models, optimize token usage, reduce LLM spending, route easy questions to cheap models, or make your AI feature more cost-effective. Also used for GPT-4 costs too much for production, AI bill keeps growing, how to reduce OpenAI costs, optimize LLM token usage, smart model routing saves money, prompt is too long and expensive, cheaper than GPT-4 with same quality.

#### `/ai-switching-models`
Switch AI providers or models without breaking things. Use when you want to switch from OpenAI to Anthropic, try a cheaper model, stop depending on one vendor, compare models side-by-side, a model update broke your outputs, you need vendor diversification, or you want to migrate to a local model. Also use when your prompt broke after a model update, prompts that work for GPT-4 do not work for Claude or Llama, or you need to do a model migration. Covers DSPy model portability with provider config, re-optimization, model comparison, and multi-model pipelines. Also used for migrate from OpenAI to Anthropic, GPT to Claude migration, try Llama instead of GPT, model comparison framework, multi-provider AI setup, avoid vendor lock-in for AI, prompts break when switching models, model-agnostic AI code.

#### `/ai-monitoring`
Know when your AI breaks in production. Use when you need to monitor AI quality, track accuracy over time, detect model degradation, set up alerts for AI failures, log predictions, measure production quality, catch when a model provider changes behavior, build an AI monitoring dashboard, or prove your AI is still working for compliance. Also use when you are seeing silent quality drops in production, a model provider changed behavior without warning, or you are dealing with prompt drift. Covers DSPy evaluation for ongoing monitoring, prediction logging, drift detection, and alerting., AI observability, LLM monitoring dashboard, model performance tracking, detect AI quality regression, production AI alerting, Datadog for AI, LLM metrics and logging, when did my AI start getting worse, AI uptime monitoring.

#### `/ai-tracing-requests`
See exactly what your AI did on a specific request. Use when you need to debug a wrong answer, trace a specific AI request, profile slow AI pipelines, find which step failed, inspect LM calls, view token usage per request, build audit trails, or understand why a customer got a bad response. Covers DSPy inspection, per-step tracing, OpenTelemetry instrumentation, and trace viewer setup., debug slow AI response, why is my AI pipeline slow, trace LLM token usage, OpenTelemetry for AI, Langfuse tracing, AI observability per request, debug wrong AI answer for specific user, which LLM call failed, latency profiling for AI, audit trail for AI decisions, inspect what the AI actually saw, per-request AI debugging, production AI request logs, DSPy inspect_history, trace AI reasoning steps.

#### `/ai-tracking-experiments`
Track which optimization experiment was best. Use when you have run multiple optimization passes, need to compare experiments, want to reproduce past results, need to pick the best prompt configuration, track experiment costs, manage optimization artifacts, decide which optimized program to deploy, or justify your choice to stakeholders. Covers experiment logging, comparison, and promotion to production., MLflow for prompt experiments, Weights and Biases for LLM, track prompt versions, experiment management for AI, which optimization run was best, A/B testing AI prompts, compare model performance across runs, version control for prompts, prompt experiment tracking, reproduce my best AI configuration, optimization history, rollback to previous prompt version, AI experiment dashboard.

#### `/ai-fixing-errors`
Fix broken AI features. Use when your AI is throwing errors, producing wrong outputs, crashing, returning garbage, not responding, or behaving unexpectedly. Also use when you get Could not parse LLM output errors, DSPy program crashes, LLM timeout or rate limit errors, API key not working with DSPy, JSON parse error from LLM, model returns empty response, AI works sometimes but fails other times, intermittent LLM failures, debug DSPy pipeline, context window exceeded, token limit error, AI feature stopped working overnight, production AI errors.

### Meta

#### `/ai-request-skill`
Request or contribute a new AI skill that does not exist yet.

## API-first skills (`dspy-` prefix)

For users who already know DSPy and want help with a specific concept or tool.

### Core concepts

#### `/dspy-signatures`
Use when you need to define the input/output contract for an LM call — choosing between inline and class-based signatures, adding type constraints, or using Pydantic models for structured outputs. Common scenarios - defining input and output fields for an LM call, adding type constraints to outputs, using Pydantic models for complex structured output, choosing between inline string signatures and class-based signatures, or declaring field descriptions that guide the model. Related - ai-parsing-data, ai-following-rules, dspy-predict, dspy-modules. Also used for dspy.Signature, dspy.InputField, dspy.OutputField, define LM call interface, typed outputs in DSPy, Pydantic model as signature, inline vs class signature, field descriptions in DSPy, structured output schema, input output contract for LLM, how to define DSPy signature, type hints in signatures, class-based signature DSPy.

#### `/dspy-lm`
Use when you need to configure which language model DSPy uses — setting up providers, API keys, model parameters, or assigning different models to different pipeline stages. Common scenarios - setting up OpenAI or Anthropic API keys, configuring model parameters like temperature and max_tokens, using different models for different pipeline stages, switching between providers, using local models with Ollama or vLLM, or setting up Azure OpenAI. Related - ai-switching-models, ai-cutting-costs, ai-kickoff. Also used for dspy.LM, dspy.configure, configure language model in DSPy, OpenAI API key setup DSPy, Anthropic Claude with DSPy, use Ollama with DSPy, local model DSPy, Azure OpenAI DSPy setup, model temperature and max_tokens, different models per module, multi-model DSPy pipeline, vLLM with DSPy, change provider without changing code, model configuration DSPy.

#### `/dspy-modules`
Use when you need to compose multiple DSPy calls into a pipeline — structuring multi-step programs as reusable, optimizable components with forward() logic. Common scenarios - building a multi-step pipeline as a class, composing Predict and ChainOfThought calls in sequence, creating reusable AI components, structuring a RAG pipeline as a module, or building nested programs where one module calls another. Related - ai-building-pipelines, dspy-predict, dspy-chain-of-thought. Also used for dspy.Module, forward() method, custom DSPy module, compose DSPy calls, multi-step DSPy program, pipeline as a class, reusable AI components, nested DSPy modules, module design patterns, how to structure a DSPy program, class-based DSPy pipeline, self.predict in forward, modular AI pipeline, build complex DSPy programs, combine multiple DSPy calls into one module.

#### `/dspy-data`
Use when you need to prepare training/dev data for DSPy optimizers — loading from CSV/JSON/HuggingFace, creating Examples, setting input keys, or building train/dev splits. Common scenarios - loading a CSV of labeled examples for optimization, converting HuggingFace datasets to DSPy format, creating train/dev/test splits, building Examples with proper input keys, converting JSON data for DSPy, or preparing evaluation datasets. Related - ai-generating-data, dspy-evaluate. Also used for dspy.Example, dspy.Dataset, load training data for DSPy, CSV to DSPy examples, HuggingFace dataset in DSPy, prepare data for optimization, input_keys in DSPy, train dev split for DSPy, how to format data for DSPy optimizer, labeled examples format, create examples from JSON, what format does DSPy expect, dataset preparation for DSPy, with_inputs in DSPy Example, build evaluation dataset.

#### `/dspy-evaluate`
Use when you need to measure how well your DSPy program performs — writing metrics, scoring against a dev set, or comparing before/after optimization. Common scenarios - measuring accuracy before and after optimization, writing custom metrics for your task, scoring a program against a held-out dev set, comparing two prompt strategies, building a test suite for AI quality, or running regression tests on AI outputs. Related - ai-improving-accuracy, ai-scoring, ai-monitoring. Also used for dspy.Evaluate, dspy.evaluate, write DSPy metric function, measure AI accuracy, evaluate DSPy program, dev set evaluation, before and after optimization comparison, custom scoring function, test AI quality systematically, AI regression testing, metric-driven development, how to know if my DSPy program improved, score predictions against labels, evaluation harness for LLM, CI/CD for AI quality.

#### `/dspy-assertions`
REMOVED IN DSPy 3.x -- use dspy.Refine or dspy.BestOfN instead (see /dspy-refine, /dspy-best-of-n). Legacy documentation for dspy.Assert and dspy.Suggest kept for existing codebases only. For new code, use dspy.Refine (iterative improvement with feedback) or dspy.BestOfN (sampling, pick best). Also used for dspy.Assert, dspy.Suggest, runtime validation for LLM output, retry on bad output, backtracking on constraint violation, guard rails in DSPy.

### Prompting strategies (modules)

#### `/dspy-predict`
Use when the mapping from input to output is straightforward and does not need reasoning steps — simple classification, extraction, formatting, or Q&A where minimal latency matters. Common scenarios - simple classification tasks, basic extraction, format conversion, straightforward Q&A, or any task that does not benefit from chain-of-thought reasoning — when you want the fastest possible LM call. Related - ai-sorting, ai-parsing-data, dspy-chain-of-thought. Also used for dspy.Predict, simplest DSPy module, basic LM call in DSPy, direct prediction no reasoning, when to use Predict vs ChainOfThought, fast classification with DSPy, minimal latency LM call, simple input-output mapping, Predict vs ChainOfThought, zero overhead DSPy call, straightforward text generation, quick extraction without reasoning, one-shot prediction, basic DSPy hello world.

#### `/dspy-chain-of-thought`
Use when the task benefits from intermediate reasoning before producing an answer — multi-step logic, analysis, math, or complex classification where direct prediction fails. Common scenarios - classification tasks where the model needs to reason about edge cases, math word problems, multi-step analysis, complex question answering, legal or medical reasoning, any task where thinking before answering improves quality. Related - ai-reasoning, dspy-predict, dspy-multi-chain-comparison. Also used for dspy.ChainOfThought, CoT prompting in DSPy, think step by step, show your reasoning, intermediate reasoning steps, LLM gives wrong answer without thinking, reasoning before output, make AI explain its logic, step-by-step problem solving, when to use ChainOfThought vs Predict, add reasoning to any DSPy module, let the model think, chain of thought for classification.

#### `/dspy-program-of-thought`
Use when the task requires precise computation, math, or data manipulation — the LM writes Python code that executes in a sandbox instead of reasoning in natural language. Common scenarios - math word problems, data manipulation tasks, precise calculations the LLM gets wrong in natural language, statistical analysis, or any task where writing and executing code gives better results than reasoning in text. Related - ai-reasoning, dspy-chain-of-thought, dspy-codeact. Also used for dspy.ProgramOfThought, LLM writes code to solve problem, code generation for computation, math with LLM via code, execute Python to get answer, when chain of thought gives wrong math, computation via code not text, precise calculations with LLM, data analysis by generating code, sandbox code execution, code-based reasoning, ProgramOfThought vs ChainOfThought, solve with code not words.

#### `/dspy-react`
Use when the task requires calling external tools or APIs to gather information — multi-step tool use with reasoning, like searching databases, calling APIs, or combining multiple data sources. Common scenarios - building agents that search the web and synthesize results, multi-step information gathering from APIs, chatbots that look up data before answering, question answering that requires external knowledge, or any task needing interleaved reasoning and action. Related - ai-taking-actions, ai-searching-docs, dspy-codeact, dspy-tools. Also used for dspy.ReAct, ReAct agent pattern, reasoning and acting loop, tool-using agent in DSPy, search then answer pattern, agent with tools, multi-step tool use, interleave thinking and acting, API-calling agent, agent that reasons about tool outputs, when to use ReAct vs CodeAct, build intelligent agent with DSPy.

#### `/dspy-codeact`
Use when the agent task is best solved by writing and executing Python code — data manipulation, computation, file processing, or tasks where code is more reliable than natural language reasoning. Common scenarios - data analysis tasks where the agent writes pandas code, computation-heavy tasks where natural language reasoning fails, file processing automation, tasks requiring precise calculations, or building agents that manipulate data programmatically. Related - ai-taking-actions, dspy-react, dspy-program-of-thought. Also used for dspy.CodeAct, agent writes and runs Python code, code execution agent, data analysis agent, AI agent that writes code, computation with LLM agent, pandas automation with AI, agent that processes files, code-generating agent, execute Python in sandbox, when ReAct is not precise enough use code, programmatic problem solving agent.

#### `/dspy-multi-chain-comparison`
Use when you want higher accuracy by generating multiple reasoning chains and selecting the best answer — trading speed for quality on critical outputs. Common scenarios - high-stakes decisions where you want multiple reasoning paths compared, classification tasks where one chain of thought is not reliable enough, improving accuracy by generating several answers and selecting the best-reasoned one, or tasks where different reasoning approaches yield different answers. Related - ai-reasoning, ai-improving-accuracy, dspy-chain-of-thought. Also used for dspy.MultiChainComparison, compare multiple reasoning chains, select best reasoning path, multi-path reasoning, vote across chain-of-thought outputs, more reliable than single CoT, deliberation for hard problems, when one reasoning chain is not enough, robust reasoning through comparison, ensemble reasoning, trade speed for accuracy on critical tasks.

#### `/dspy-best-of-n`
Use when output quality varies across runs and you want to sample multiple completions and pick the best — trading latency for reliability on high-stakes outputs. Common scenarios - generating multiple candidate answers and picking the highest-scoring one, improving reliability on high-stakes classification, reducing variance in creative generation, getting better summaries by sampling several and selecting the best, or trading latency for quality on critical decisions. Related - ai-improving-accuracy, ai-making-consistent. Also used for sample multiple completions, pick the best of several LLM outputs, majority voting for LLM, self-consistency decoding, reduce LLM output variance, generate and select pattern, best candidate selection, how to make AI more reliable by trying multiple times, brute force better quality, retry and pick best, dspy.BestOfN, quality vs latency tradeoff, n=5 completions pick best.

#### `/dspy-refine`
Iterative self-improvement with dspy.Refine -- wraps any module, scores each attempt with a reward function, generates feedback on failures, and retries until a quality threshold is met. Use when you want outputs to improve through self-critique, need iterative revision of drafts, or want the LM to learn from its own mistakes within a single request. Also used for self-critique and revise, iterative improvement loop, generate then evaluate then fix, AI self-editing, multi-draft generation, revise until good enough, critique-driven refinement, when first draft is not good enough.

#### `/dspy-parallel`
Use when you have independent LM calls that can run concurrently — batch processing, fan-out patterns, or speeding up pipelines with no data dependencies between steps. Common scenarios - processing a batch of inputs through a DSPy module concurrently, fan-out patterns where multiple independent LM calls run at once, speeding up evaluation by parallelizing predictions, or reducing wall-clock time for pipelines with no data dependencies. Related - ai-building-pipelines, ai-serving-apis. Also used for dspy.Parallel, concurrent LM calls, batch processing in DSPy, parallel DSPy execution, speed up DSPy pipeline, fan-out LM calls, concurrent predictions, parallelize evaluation, async DSPy calls, reduce latency with parallel execution, batch inference DSPy, process multiple inputs at once, throughput optimization, run DSPy modules concurrently, parallel map over inputs.

#### `/dspy-rlm`
Recursive Language Model (dspy.RLM) that explores large contexts via a sandboxed Python REPL -- the LM writes code, queries sub-LMs, and iterates until it produces a final answer. Use when your input is too large for the context window, the model needs to explore data iteratively, you need recursive self-refinement with code execution, or you have research-style tasks requiring programmatic investigation. Also used for recursive language model, iterative exploration with LLM, model explores data in REPL, agent that keeps digging until it finds the answer, REPL-based reasoning, explore then answer pattern, deep research agent, when one pass is not enough.

### Optimizers

#### `/dspy-bootstrap-few-shot`
Use when you have 50+ labeled examples and want a quick accuracy boost as your first optimization step — the simplest and fastest DSPy optimizer. Common scenarios - your first optimization attempt on a new DSPy program, adding few-shot examples automatically from labeled data, quick accuracy boost before trying heavier optimizers, bootstrapping demonstrations from a teacher model, or getting started with DSPy optimization. Related - ai-improving-accuracy, dspy-labeled-few-shot. Also used for dspy.BootstrapFewShot, simplest DSPy optimizer, first optimizer to try, automatic few-shot example selection, bootstrap demonstrations from labels, quick optimization baseline, add examples to prompt automatically, teacher bootstrapping, labeled data to few-shot demos, starting point for DSPy optimization, easy accuracy improvement, how to optimize DSPy program for the first time.

#### `/dspy-bootstrap-rs`
Use when basic BootstrapFewShot is not enough and you want to search over multiple candidate demo sets — better results at the cost of more LM calls. Common scenarios - BootstrapFewShot alone is not reaching target accuracy, you want to search over multiple candidate demo sets and pick the best, optimizing for tasks where example selection matters a lot, or when you have compute budget for a more thorough search. Related - ai-improving-accuracy, dspy-bootstrap-few-shot. Also used for dspy.BootstrapFewShotWithRandomSearch, random search over demonstrations, better than basic BootstrapFewShot, search for optimal few-shot examples, brute force demo selection, try many demo combinations, more compute for better demos, upgrade from BootstrapFewShot, intermediate optimizer between simple and MIPROv2, when basic few-shot optimization is not enough, explore demonstration space.

#### `/dspy-miprov2`
Use when you want the highest-quality prompt optimization DSPy offers — jointly optimizes instructions and few-shot demos, with auto=light/medium/heavy presets. Common scenarios - you want the best possible accuracy from prompt optimization, jointly tuning instructions and few-shot demonstrations, using auto presets for different compute budgets, or when COPRO or BootstrapFewShot alone are not reaching your accuracy target. Related - ai-improving-accuracy, dspy-copro, dspy-bootstrap-few-shot. Also used for dspy.MIPROv2, best DSPy optimizer, highest quality optimization, auto=light medium heavy, joint instruction and demo optimization, most powerful prompt optimizer, MIPROv2 vs COPRO vs BootstrapFewShot, which optimizer should I use, state of the art prompt optimization, when to use MIPROv2, optimize both instructions and examples, heavy optimization for production, best optimizer for accuracy.

#### `/dspy-gepa`
Use when you want to optimize instructions without few-shot examples — a lightweight alternative to COPRO when you do not have or do not want to use demonstrations. Common scenarios - optimizing instructions when you do not have or do not want to use few-shot demonstrations, lightweight instruction search as a first step, tasks where examples in the prompt confuse the model, or when you want fast instruction optimization without the cost of COPRO. Related - ai-improving-accuracy, dspy-copro, dspy-miprov2. Also used for dspy.GEPA, instruction optimization without demos, lightweight prompt optimization, optimize instructions only, no few-shot examples needed, GEPA vs COPRO, quick instruction search, when demonstrations hurt performance, zero-shot optimization, instruction-only optimizer, simplest instruction tuner, fast prompt optimization, skip few-shot and just tune instructions, optimize Pydantic field descriptions, GEPA structured output, GEPA does not optimize field desc.

#### `/dspy-copro`
Use when you want to optimize instructions by generating many candidates and picking the best — useful when few-shot demos alone are not enough and you want to tune the task description itself. Common scenarios - your current task instructions produce mediocre results, you want to automatically generate and test many instruction variants, the task is hard to describe in one sentence, or few-shot examples alone are not improving quality enough. Related - ai-improving-accuracy, dspy-gepa, dspy-miprov2. Also used for dspy.COPRO, instruction optimization, optimize task description, generate better prompts automatically, prompt engineering automation, find the best instruction for my task, automatic prompt generation, instruction tuning without fine-tuning, COPRO vs MIPROv2, when to optimize instructions vs demos, instruction search, prompt optimization by generating candidates, systematic prompt improvement.

#### `/dspy-simba`
Use when you want conservative, incremental optimization — making small targeted improvements rather than large changes, useful for already-working programs that need fine-tuning. Common scenarios - your program already works well and you want to improve it without breaking what works, conservative optimization that preserves existing quality, fine-tuning a production program incrementally, or when aggressive optimization causes regressions. Related - ai-improving-accuracy, dspy-miprov2, dspy-refine. Also used for dspy.SIMBA, conservative optimization, incremental improvement, do not break what works, small targeted optimization, safe optimization for production, avoid regressions during optimization, production-safe optimizer, gentle optimization, when MIPROv2 changes too much, preserve existing quality, stable optimization, risk-averse prompt tuning, optimize without regressions.

#### `/dspy-better-together`
Use when you have already tried prompt-only optimization and want the next level — jointly tuning prompts and model weights for maximum quality. Common scenarios - you have maxed out prompt optimization and need the next level, combining instruction tuning with weight tuning for maximum quality, making a small model match a large model through joint optimization, or squeezing the last few percent of accuracy. Related - ai-fine-tuning, ai-improving-accuracy, ai-cutting-costs. Also used for dspy.BetterTogether, joint prompt and weight optimization, beyond prompt engineering, combine fine-tuning with prompt optimization, maximum possible quality from DSPy, hybrid optimization strategy, prompt optimization hit a ceiling, fine-tune and optimize prompts at the same time, advanced DSPy optimization, best possible accuracy, what to try after MIPROv2, next level AI quality.

#### `/dspy-bootstrap-finetune`
Use when you need maximum quality from a smaller/cheaper model — generates training data from a teacher model and fine-tunes a student model weights. Common scenarios - distilling GPT-4 quality into a cheaper model, generating training data from a strong teacher to fine-tune a weak student, reducing inference costs by replacing an expensive model with a fine-tuned small one, or building a production model that is fast and cheap. Related - ai-fine-tuning, ai-cutting-costs, dspy-better-together. Also used for dspy.BootstrapFinetune, model distillation with DSPy, teacher-student training, fine-tune small model from GPT-4 outputs, reduce API costs with fine-tuning, generate training data then fine-tune, cheap model same quality, distill large model into small model, fine-tune Llama from GPT-4, production model training, move from API to self-hosted model.

#### `/dspy-ensemble`
Use when you have multiple optimized versions of a program and want to combine them — voting, averaging, or routing across program variants for more robust outputs. Common scenarios - you have optimized several versions of a program and want to combine the best ones, using majority voting across multiple programs for higher accuracy, building a robust system by routing to different specialized programs, or reducing variance by averaging outputs. Related - ai-improving-accuracy, ai-making-consistent, dspy-bootstrap-rs. Also used for dspy.Ensemble, combine multiple optimized programs, majority voting across models, ensemble of DSPy programs, voting for reliability, reduce variance with multiple programs, aggregate predictions, combine outputs from different optimizers, when one program is not reliable enough, model committee, ensemble for production robustness, multiple programs one answer.

#### `/dspy-infer-rules`
Use when you want to extract interpretable decision logic from labeled examples — generating explicit rules that explain patterns in your data. Common scenarios - extracting business rules from labeled classification examples, understanding why a model makes certain predictions, generating human-readable decision criteria from data, building interpretable classifiers, or documenting implicit labeling logic from annotators. Related - ai-following-rules, ai-sorting. Also used for dspy.InferRules, extract rules from examples, interpretable AI decisions, understand classification logic, generate decision rules from labels, explainable AI with DSPy, turn labeled data into explicit rules, human-readable classification rules, rule extraction from training data, when you need to explain why AI decided, interpretable model logic, audit AI decision process, regulatory compliance explainability, extract patterns from labeled data.

#### `/dspy-knn-few-shot`
Use when you want few-shot demos that are dynamically selected per input based on similarity — better than fixed demos when inputs vary widely. Common scenarios - inputs vary widely and fixed examples do not cover enough cases, dynamically selecting the most relevant demos per input, building a retrieval-augmented prompt with similar examples, or when static few-shot examples work for some inputs but fail on others. Related - ai-improving-accuracy, dspy-labeled-few-shot, dspy-bootstrap-few-shot. Also used for dspy.KNNFewShot, dynamic few-shot selection, similar examples per input, retrieval-augmented few-shot, adaptive demonstrations, nearest neighbor example selection, dynamic prompt construction, different examples for different inputs, embedding-based demo retrieval, when fixed examples do not generalize, per-input demo selection, contextual few-shot examples, smart example selection.

#### `/dspy-labeled-few-shot`
Use when you have hand-picked high-quality examples and want to use them directly as few-shot demonstrations — no bootstrapping, just your curated demos. Common scenarios - you have expert-curated examples that you trust more than bootstrapped ones, hand-picked demonstrations for high-stakes tasks, using existing labeled data directly without bootstrapping, or when you want full control over which examples appear in the prompt. Related - dspy-bootstrap-few-shot, dspy-knn-few-shot, ai-generating-data. Also used for dspy.LabeledFewShot, hand-picked examples in prompt, curated demonstrations, use my own examples directly, manual few-shot setup, expert-labeled demonstrations, no bootstrapping just my examples, static few-shot with labeled data, gold standard examples, when you trust your examples more than auto-generated ones, controlled few-shot demos, fixed example set in prompt.

### Infrastructure

#### `/dspy-adapters`
Use when you need to customize how DSPy formats prompts for a specific provider — switching from chat to completion format, forcing JSON output, or debugging prompt rendering issues. Common scenarios - debugging why your prompt looks wrong when sent to the model, switching from OpenAI to Anthropic and the formatting breaks, forcing the model to return valid JSON instead of markdown, working with completion-style models that do not support chat format, customizing system messages, or handling models that choke on structured output instructions. Related - ai-switching-models, ai-following-rules, ai-parsing-data. Also used for prompt template rendering, how DSPy builds the prompt, custom system message in DSPy, JSON mode not working, model ignores format instructions, switch from chat to completion API, dspy.ChatAdapter, dspy.JSONAdapter, prompt formatting issues, debug what DSPy sends to the model, dspy.XMLAdapter, XML output format.

#### `/dspy-chatadapter`
Deep dive into dspy.ChatAdapter -- the default adapter that formats DSPy signatures into multi-turn chat messages with field delimiters, parses LM responses back into typed Python objects, and falls back to JSONAdapter on failure. Use when you need to understand how DSPy builds prompts, debug why a model ignores output format, customize prompt rendering, enable native function calling, use callbacks, generate fine-tuning data, or control the JSON fallback. Also used for how DSPy formats prompts, field delimiters, prompt template rendering, parse error debugging, ChatAdapter vs JSONAdapter vs TwoStepAdapter, format_finetune_data, dspy prompt inspection, why model output is wrong format, adapter callbacks, native function calling in DSPy.

#### `/dspy-tools`
Use when you need to give DSPy agents tool-calling abilities — wrapping Python functions as tools, building tool-using pipelines, or setting up code execution environments. Common scenarios - wrapping a Python function as a tool for DSPy agents, building tool-using pipelines, setting up a calculator or search tool, giving agents access to databases or APIs, or configuring code execution environments for agents. Related - ai-taking-actions, dspy-react, dspy-codeact. Also used for dspy.Tool, wrap function as DSPy tool, give agent tools, tool calling in DSPy, function calling for agents, build custom tools for DSPy agent, calculator tool for LLM, search tool for agent, database tool for AI, Python function to agent tool, MCP tools with DSPy, tool registry, how to define tools in DSPy, agent tool configuration, executable tools for AI agents.

#### `/dspy-retrieval`
DSPy retrieval modules (dspy.Retrieve, dspy.ColBERTv2, dspy.Embedder, dspy.retrievers.Embeddings) for searching documents, computing embeddings, and building RAG pipelines. Use when you need to search over documents, build a RAG pipeline, connect DSPy to a vector database, compute embeddings for semantic search, set up ChromaDB or Pinecone with DSPy, or build knowledge-grounded question answering. Also used for RAG pipeline in DSPy, vector database integration, semantic search, embedding retrieval, retrieval augmented generation setup, connect knowledge base to DSPy, search documents then answer, grounded generation with retrieval.

#### `/dspy-primitives`
DSPy typed wrappers (dspy.Image, dspy.Audio, dspy.Code, dspy.History) for multimodal data in signatures. Use when working with non-text inputs like images, audio, or code, building multimodal AI pipelines, processing images alongside text, handling audio transcription inputs, working with code files as typed inputs, or managing conversation history in multi-turn chatbots. Also used for multimodal DSPy, image input in DSPy signature, process images with DSPy, audio input in DSPy, typed fields in signatures, non-text data in DSPy, vision model with DSPy, Claude vision with DSPy, multimodal pipeline, image classification with DSPy, pass images to language model, conversation history type, structured types beyond strings.

#### `/dspy-utils`
Use when you need DSPy infrastructure - streaming responses, caching control, debugging with inspect_history, saving/loading programs, async execution, or MCP integration. Common scenarios - enabling streaming responses in production, controlling the cache to avoid stale results, debugging with inspect_history to see raw prompts, saving and loading optimized programs, running DSPy modules asynchronously, or integrating with MCP servers. Related - ai-tracing-requests, ai-serving-apis, ai-monitoring. Also used for dspy.inspect_history, dspy.settings.configure, streaming DSPy output, cache control in DSPy, save and load DSPy program, async DSPy execution, MCP integration with DSPy, debug DSPy prompts, see what DSPy sent to the model, DSPy program serialization, production DSPy utilities, clear DSPy cache, view prompt history, async await with DSPy, stream tokens from DSPy.

### Ecosystem tools

#### `/dspy-langtrace`
Use Langtrace for DSPy observability and tracing. Use when you want to set up Langtrace, langtrace-python-sdk, auto-instrument DSPy, trace DSPy calls, LLM observability, app.langtrace.ai, or self-hosted tracing. Also used for langtrace setup, langtrace API key, pip install langtrace-python-sdk, DSPy tracing, auto-instrument DSPy, langtrace self-hosted, langtrace docker, trace LM calls, langtrace vs phoenix, langtrace cloud.

#### `/dspy-phoenix`
Use Arize Phoenix for DSPy tracing and evaluation. Use when you want to set up Phoenix, arize-phoenix, openinference, DSPyInstrumentor, open-source trace viewer, localhost:6006, or LLM evals. Also used for phoenix setup, arize phoenix, pip install arize-phoenix, phoenix local UI, phoenix evaluations, DSPy trace viewer, open-source LLM observability, phoenix vs langtrace, openinference-instrumentation-dspy, phoenix.otel register.

#### `/dspy-weave`
Use W&B Weave for DSPy experiment tracking and observability. Use when you want to set up Weave, W&B, wandb, Weights & Biases, experiment dashboard, weave.op, or team collaboration for DSPy. Also used for weave setup, pip install weave, weave.init, wandb project, W&B experiment tracking, weave decorator, weave.op decorator, wandb dashboard, compare optimization runs, team experiment tracking.

#### `/dspy-mlflow`
Use MLflow for DSPy tracing, experiment tracking, and model registry. Use when you want to set up MLflow, mlflow.dspy.autolog, MLflow Tracing, MLflow experiment tracking, MLflow model registry, or full ML lifecycle management. Also used for mlflow setup, pip install mlflow, mlflow.set_experiment, mlflow UI, mlflow model versioning, mlflow OpenTelemetry, mlflow vs wandb, mlflow tracing DSPy, register DSPy model.

#### `/dspy-langwatch`
Use LangWatch for DSPy auto-tracing and real-time optimizer progress. Use when you want to set up LangWatch, langwatch.dspy.init, auto-tracing DSPy, real-time optimization dashboard, optimizer progress tracking, app.langwatch.ai, or DSPy optimizer dashboard. Also used for langwatch setup, pip install langwatch, langwatch trace, optimizer progress, real-time optimization, watch optimizer run, LangWatch self-hosted, langwatch docker, langwatch vs langtrace, langwatch autotrack_dspy.

#### `/dspy-langfuse`
LLM observability for DSPy with Langfuse -- auto-trace every LM call, attach scores and evaluations, run annotation queues for human review, and track experiments across prompt versions. Use when you want to set up Langfuse, langfuse.com, openinference-instrumentation-dspy, trace DSPy calls, LLM observability with scores, annotation queues, or experiment tracking. Also used for langfuse setup, pip install langfuse, DSPy trace viewer, langfuse vs phoenix, langfuse vs langtrace, observe decorator with DSPy, self-hosted tracing with evaluation, production LLM monitoring with scoring.

#### `/dspy-ragas`
Use Ragas to evaluate DSPy RAG pipelines with decomposed metrics. Use when you want to evaluate RAG quality, measure faithfulness, context precision, context recall, answer relevancy, or diagnose retriever vs generator issues. Also used for ragas, pip install ragas, ragas evaluate, RAG evaluation, faithfulness metric, context precision, context recall, answer relevancy, answer correctness, decomposed RAG metrics, ragas dspy, DSPyOptimizer ragas, ragas[dspy], EvaluationDataset, ragas vs dspy.Evaluate, which RAG metric, retriever vs generator quality.

#### `/dspy-qdrant`
Use Qdrant as a vector database with DSPy, or connect any vector DB (Pinecone, ChromaDB, Weaviate) with custom retrievers. Use when you want to set up Qdrant, QdrantRM, dspy-qdrant, vector database for DSPy, vector search, hybrid search, or build custom retrievers for Pinecone, ChromaDB, or Weaviate. Also used for qdrant, dspy-qdrant, QdrantRM, vector database, vector search, pinecone DSPy, chromadb DSPy, weaviate DSPy, vector DB for DSPy, pip install dspy-qdrant, qdrant docker, qdrant cloud, hybrid search DSPy, sparse dense vectors, custom dspy.Retrieve, which vector DB for DSPy, DSPy 3.0 retriever removed.

#### `/dspy-ollama`
Run DSPy with local models using Ollama — no API key needed. Use when you want to run DSPy locally, use Ollama, set up a local LLM, run offline, or configure local model parameters. Also used for ollama, local model, run LLM locally, llama local, self-hosted LLM, ollama serve, ollama_chat, local inference, run DSPy offline, no API key needed, ollama pull, ollama list, ollama rm, num_ctx, ollama context window, ollama GPU, OLLAMA_NUM_GPU, OLLAMA_HOST, ollama remote, ollama embeddings, nomic-embed-text, dspy.Embedder ollama, which local model, best model for ollama, ollama too slow, ollama vs vllm, develop locally deploy remotely, ollama environment variables, ollama systemd, ollama background service.

#### `/dspy-vllm`
Use vLLM for high-throughput production serving of self-hosted models with DSPy. Use when you want production LLM serving, tensor parallelism, multi-GPU inference, batch processing, or high-concurrency self-hosted models. Also used for vllm, vLLM, production serving, high throughput LLM, tensor parallelism, self-hosted production, PagedAttention, local production server, GPU serving, batch inference, vllm serve, pip install vllm, multi-GPU LLM, speculative decoding, continuous batching, deploy local model, NVIDIA GPU serving, openai compatible server, AWQ quantization vllm, GPTQ vllm.

#### `/dspy-vizpy`
Use VizPy as a drop-in prompt optimizer for DSPy. Use when you want to try VizPy, vizops, ContraPromptOptimizer, PromptGradOptimizer, a commercial alternative to GEPA, a third-party prompt optimizer, or a different optimization backend. Also used for vizpy optimizer, vizpy vs GEPA, vizpy vs MIPROv2, commercial prompt optimization, ContraPrompt for classification, PromptGrad for generation, vizpy API key, pip install vizpy, vizpy free tier.

## Example multi-skill sequences

These show how to combine `ai-` and `dspy-` skills for real-world projects.

**"I want to build an AI-powered help center"**
1. `/ai-searching-docs` — Build RAG over your help articles
2. `/ai-stopping-hallucinations` — Ground answers in source docs with citations
3. `/dspy-evaluate` — Set up SemanticF1 and answer_passage_match metrics
4. `/dspy-miprov2` — Optimize prompts and demos for your best metric
5. `/ai-serving-apis` — Deploy as an API for your frontend

**"I want to auto-process incoming invoices"**
1. `/ai-parsing-data` — Extract vendor, amount, line items, dates from PDF/email text
2. `/dspy-signatures` — Define a typed Signature with Pydantic models for invoice fields
3. `/ai-checking-outputs` — Validate extracted fields (amounts add up, dates are valid)
4. `/ai-sorting` — Route to the right approval workflow based on amount/department
5. `/dspy-bootstrap-few-shot` — Auto-generate demos from your labeled invoices

**"I need a support ticket system with AI triage"**
1. `/ai-sorting` — Classify tickets by category and priority
2. `/ai-summarizing` — Generate a one-line summary for the queue
3. `/dspy-modules` — Compose classify + summarize into a single Module
4. `/dspy-evaluate` — Measure end-to-end pipeline quality
5. `/dspy-miprov2` — Optimize the full pipeline

**"Build a content moderation system for our app"**
1. `/ai-moderating-content` — Build the base classifier with severity levels
2. `/ai-following-rules` — Enforce your content policy rules strictly
3. `/ai-testing-safety` — Red-team it to find bypasses
4. `/dspy-best-of-n` — Run moderation N times and pick the most conservative result
5. `/ai-monitoring` — Track moderation quality in production

**"I want to replace our expensive GPT-4 system with something cheaper"**
1. `/dspy-evaluate` — Measure current quality as a baseline with proper metrics
2. `/dspy-bootstrap-finetune` — Generate training data from your best GPT-4 outputs
3. `/ai-fine-tuning` — Fine-tune a cheap model on that data
4. `/dspy-lm` — Swap to the fine-tuned model with fallback to GPT-4
5. `/ai-monitoring` — Track quality after the switch

**"Build an AI research assistant that finds and summarizes papers"**
1. `/dspy-retrieval` — Set up ColBERTv2 or embeddings over your paper corpus
2. `/ai-summarizing` — Summarize retrieved papers
3. `/dspy-react` — Build an agent that searches, retrieves, and summarizes in a loop
4. `/dspy-tools` — Wrap external APIs (arxiv, semantic scholar) as DSPy tools
5. `/ai-coordinating-agents` — Orchestrate multiple specialist agents

**"I need AI to grade student essays against a rubric"**
1. `/ai-scoring` — Build rubric-based scoring with per-criteria grades
2. `/dspy-chain-of-thought` — Add reasoning so the grader explains its scores
3. `/ai-making-consistent` — Ensure grading is fair and repeatable across essays
4. `/dspy-evaluate` — Measure agreement with teacher-graded examples
5. `/dspy-miprov2` — Optimize grading prompts against teacher labels

**"We need a chatbot that can look up orders and process returns"**
1. `/ai-building-chatbots` — Build the conversational interface with memory
2. `/dspy-tools` — Wrap order lookup, return processing, status checks as tools
3. `/dspy-react` — Wire the tools into a ReAct agent that reasons about what to call
4. `/ai-following-rules` — Enforce return policy rules (time limits, conditions)
5. `/ai-testing-safety` — Test for prompt injection and policy bypass

**"I want to monitor our AI in production and catch when it degrades"**
1. `/dspy-evaluate` — Define metrics and build an evaluation suite
2. `/ai-monitoring` — Set up production quality tracking and alerts
3. `/dspy-utils` — Add inspect_history and StreamListener for debugging
4. `/ai-tracing-requests` — Add request-level tracing for debugging failures
5. `/ai-tracking-experiments` — Track optimization runs when you need to fix issues
