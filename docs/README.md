# Docs

How the work is documented, so it can be followed step by step.

| Where | What | When it's written |
|---|---|---|
| `../CHANGELOG.md` | One line per change, newest first | Every change |
| `devlog/YYYY-MM-DD-<topic>.md` | Work-session report: what changed, why, how it works, how it was verified, deviations from the paper, next steps | Every work session |
| `architecture.md` | System model: block diagram (B1…B8), interface table, RL episode sequence/state diagrams (Mermaid) | Whenever structure or interfaces change |
| `adr/NNNN-<decision>.md` | Architecture Decision Record: one design choice, the alternatives, and the consequences | When a choice is hard to reverse or not obvious |
| `quiz/YYYY-MM-DD-quiz-NN.md` | 5–8 concept questions (block roles, data flow, design reasons, paper deviations), answers in collapsed callouts | After each round of docs |

Paper section references (e.g. "Sec. 6", "Tab. 1") refer to *ReActor: Reinforcement Learning for Physics-Aware Motion Retargeting*.

Each devlog names the blocks it changed (B1…B8) so it can be traced on the architecture diagram.
