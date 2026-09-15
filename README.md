# Hiver Support Agent

An end-to-end customer-support agent built from historical Twitter support conversations. The system classifies customer intent, decides whether escalation is appropriate, retrieves similar historical support interactions, and generates a grounded support reply using a local Qwen3 model.

## Problem

The goal is to assist support teams by:

- identifying the customer's support intent
- determining whether the issue should be escalated
- retrieving relevant historical support interactions
- generating a concise response grounded in previous support behavior

The system is designed to avoid relying on unsupported information and to use historical support conversations as the main source of response guidance.

## Dataset

The project uses the Twitter Customer Support dataset.

The processed Apple Support subset contains approximately 106K customer-support messages. A conversation-level dataset was constructed from customer messages and corresponding Apple Support responses.

The final conversation dataset contains 98,575 cleaned customer-response pairs before splitting.

## Intent Taxonomy

The project uses 13 support intents:

1. battery_issue
2. ios_update_issue
3. performance_freezing
4. restart_boot_issue
5. charging_issue
6. audio_call_issue
7. display_touch_issue
8. network_connectivity
9. account_icloud_issue
10. apps_media_issue
11. purchase_order_support
12. how_to_feature_question
13. other_unclear

A manually labeled set of 50 examples was created to guide the semantic intent classifier.

## Data Processing

The data pipeline includes:

- identifying Apple Support customer messages
- pairing customer messages with recorded Apple responses
- removing empty or invalid examples
- removing duplicate customer-response pairs
- constructing conversation/thread data
- performing train/validation/test splits
- checking for leakage across splits

The final thread-aware split contains:

- Train: 78,866 rows
- Validation: 9,859 rows
- Test: 9,850 rows

Thread overlap across train, validation, and test sets was checked and found to be zero.

## Intent Classification

Several baselines were evaluated.

### Majority-class baseline

Accuracy: 13.5%

### TF-IDF + Logistic Regression

Accuracy: 24.5%

### Semantic classification

A Sentence-Transformer embedding model was used with the manually labeled examples.

This approach substantially improves over the simple majority and TF-IDF baselines, although the 13-class problem remains challenging because several intents are semantically similar.

## Retrieval

Historical Apple Support conversations are embedded using:

`all-MiniLM-L6-v2`

For a new customer message, the system retrieves the three most semantically similar historical customer-support interactions.

The retrieved examples provide evidence about how Apple historically responded to similar issues.

Example:

Customer:

> My iPhone battery is draining very quickly after the latest update.

The system retrieves historical examples involving battery drain after iOS updates and uses their responses as generation evidence.

## Response Generation

The response generator uses:

`Qwen3 4B`

through Ollama running locally.

No paid LLM API is required.

The prompt instructs the model to:

- use retrieved historical examples as evidence
- remain concise and professional
- ask for missing information when necessary
- avoid inventing links, policies, prices, guarantees, diagnoses, or unsupported technical facts
- return only the customer-facing response

A small output-cleaning layer removes accidental Qwen reasoning/thinking blocks before displaying the response.

## Escalation

A rule-based escalation policy was implemented using signals such as:

- explicit requests for a human, manager, or escalation
- repeated unsuccessful troubleshooting
- severe device failures
- account/security issues
- purchase/payment issues
- prolonged unresolved problems

Evaluation on the frozen 200-example set produced:

| Metric | Score |
|---|---:|
| Accuracy | 54.5% |
| Precision | 44.4% |
| Recall | 23.3% |
| F1 | 30.5% |

The main limitation is low recall: the current rules miss some cases that require escalation. This is an identified area for future improvement.

## Frozen Evaluation Set

A frozen 200-example golden set was created for final evaluation.

It contains manually assigned:

- intent labels
- escalation labels

The frozen set is kept separate from the training/retrieval data to avoid evaluation leakage.

## End-to-End Architecture

```text
Customer message
       |
       v
Semantic intent classifier
       |
       v
Escalation policy
       |
       v
Historical conversation retrieval
       |
       v
Qwen3 4B local generation
       |
       v
Clean customer-facing reply