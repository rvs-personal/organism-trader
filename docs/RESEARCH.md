# Research plan

## Main question

Can a fixed language model make better financial decisions when the system supplies structured memory, rules and reviewed lessons?

A second question concerns cost. Does each added feature improve results enough to justify its inference time and token use?

## Existing work

The broad method already exists. This project must compare its results with earlier work.

- [Reflexion](https://arxiv.org/abs/2303.11366): feedback and text memory without model-weight updates.
- [FinMem](https://arxiv.org/abs/2311.13743): trading agents with layered memory and character settings.
- [FinCon](https://arxiv.org/abs/2407.06567): cooperating financial agents with revised investment beliefs.
- [Agentic Context Engineering](https://arxiv.org/abs/2510.04618): structured updates to an agent's context.
- [StockBench](https://github.com/ChenYXxxx/stockbench): an external trading evaluation environment.

These links describe related work. This repository does not contain a verified reproduction of those results.

## Planned comparisons

Keep the base model, market information, execution costs and inference budget comparable.

1. Run the model with a basic decision procedure.
2. Add explicit rules.
3. Add recent factual memory.
4. Add unfiltered reflection in a separate condition.
5. Compare this with reviewed memory.
6. Test pressure wording and cooperating agents as separate conditions.
7. Test transfer to another model and unseen market periods.

The first four agent styles are trend, contrarian, risk-aware and experimental. Four paired agents receive performance-pressure wording. This wording is an experimental treatment. It is not evidence that the model has feelings or that pressure improves results.

## Memory terms

- A belief is a claim the agent considers true.
- A hypothesis is an idea to test.
- A reflection can propose beliefs, hypotheses and lessons.
- Reviewed memory contains accepted observations and guidance.

In the governed workflow, reflection proposals do not automatically become active policy. An external review can accept or reject them. This makes the review process part of the tested system.

## Evaluation procedure

Separate development data from evaluation data before the final test. Do not select evaluation periods because their outcomes are already known.

Freeze the decision procedure and memory-update procedure before evaluation. Memory can update during the test only through that fixed procedure. Record all changes and all external review calls.

Use repeated runs. Report returns, drawdown, exposure, trading costs, failures, tokens and latency. Report negative results as well as positive results.

Use the same dates and execution assumptions for each comparison. Do not compare an NSE return directly with a published US-stock result.

## Known limits

The development runs used manual reviews and several software changes. Those runs are exploratory evidence. They do not establish the causal effect of one feature.

Generic rule wording does not prevent overfitting. Lessons must work on data that did not produce them.

The public source supports daily simulation. It has no live brokerage execution. A news feed is not included. The company-event tool currently returns no events.

The public demo uses synthetic data and a stub. No performance findings from that demo should be presented as AI investment results.
