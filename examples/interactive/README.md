# Interactive embedding example

This command demonstrates three interactive Cogito paths in one conversation:

1. `QuestionRegistry` publishes a structured `ask_user` question, while the
   CLI uses `Pending` and `Answer` to release the blocked tool call.
2. `EnableAutoPlan` and `WithPlanApproval` accept, reject, edit, or revise an
   automatically generated plan before its local tool runs.
3. A named background agent runs through `WithAgentDispatcher`. The dispatcher
   waits for `WithOnPark`, returns a report, and Cogito injects that completion
   into the parent so `WithOnResume` and the final reply observe the real
   park/resume path.

Run it interactively:

```sh
go run ./examples/interactive
```

Or run the deterministic happy path:

```sh
printf 'print commands\napprove\n' | go run ./examples/interactive
```

The first prompt accepts an option label, comma-separated labels, or
`text:YOUR ANSWER`. The plan prompt accepts:

- `approve`
- `reject`
- `feedback:TEXT` to request another proposal
- `edit:DESCRIPTION | STEP [| STEP]` to approve a replacement plan

The `scriptedModel` and dispatcher are explicit offline stand-ins for a real
LLM client and worker service. They inspect the conversation and requested JSON
schema rather than relying on a fragile request count. No API key or network is
used.

The CLI owns one stdin reader goroutine. EOF and Ctrl-C cancel pending work;
blocking readers supplied to `run` must implement `io.Closer` so cancellation
can join that reader. The message-injection channel remains open until Cogito
returns, and registered background agents are canceled and joined before the
example exits.
