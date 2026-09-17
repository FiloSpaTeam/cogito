package cogito

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/sashabaranov/go-openai"
)

type streamTestResult struct {
	result string
	err    error
}

type streamTestRunner struct {
	mu      sync.Mutex
	results []streamTestResult
	calls   int
	started chan<- struct{}
	waitFor <-chan struct{}
	waitCtx context.Context
}

func (r *streamTestRunner) Run(map[string]any) (string, any, error) {
	if r.started != nil {
		close(r.started)
	}
	if r.waitFor != nil {
		if r.waitCtx == nil {
			return "", nil, errors.New("stream test runner missing wait context")
		}
		select {
		case <-r.waitFor:
		case <-r.waitCtx.Done():
			return "", nil, fmt.Errorf("stream test runner wait: %w", r.waitCtx.Err())
		}
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	i := r.calls
	r.calls++
	if i >= len(r.results) {
		return "", nil, fmt.Errorf("unexpected call %d", i+1)
	}
	result := r.results[i]
	return result.result, nil, result.err
}

const streamTestTimeout = 2 * time.Second

func streamTestContext(t *testing.T) context.Context {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), streamTestTimeout)
	t.Cleanup(cancel)
	return ctx
}

func awaitStreamTest[T any](t *testing.T, ctx context.Context, ch <-chan T, label string) T {
	t.Helper()
	select {
	case value := <-ch:
		return value
	case <-ctx.Done():
		t.Fatalf("timed out waiting for %s: %v", label, ctx.Err())
		var zero T
		return zero
	}
}

func awaitStreamAgent(t *testing.T, ctx context.Context, agent *AgentState) {
	t.Helper()
	select {
	case <-agent.done:
	case <-ctx.Done():
		t.Fatalf("timed out waiting for agent %s: %v", agent.ID, ctx.Err())
	}
}

func (r *streamTestRunner) callCount() int {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.calls
}

func streamTestTool(name string, runner *streamTestRunner) ToolDefinitionInterface {
	return NewToolDefinition[map[string]any](runner, map[string]any{
		"type":       "object",
		"properties": map[string]any{},
	}, name, name)
}

type eventRecorder struct {
	mu     sync.Mutex
	events []StreamEvent
}

func (r *eventRecorder) record(ev StreamEvent) {
	r.mu.Lock()
	r.events = append(r.events, ev)
	r.mu.Unlock()
}

func (r *eventRecorder) matching(match func(StreamEvent) bool) []StreamEvent {
	r.mu.Lock()
	defer r.mu.Unlock()
	var events []StreamEvent
	for _, ev := range r.events {
		if match(ev) {
			events = append(events, ev)
		}
	}
	return events
}

func TestStreamToolResultSequentialSuccessAndDedicatedCallback(t *testing.T) {
	runner := &streamTestRunner{results: []streamTestResult{{result: "ok"}}}
	tool := streamTestTool("echo", runner)
	llm := newSequenceLLM(toolTurn("echo", `{}`))
	var recorder eventRecorder
	var dedicated atomic.Int32

	_, err := ExecuteTools(llm, NewEmptyFragment().AddMessage(UserMessageRole, "run"),
		WithTools(tool), DisableSinkState, WithIterations(1),
		WithStreamCallback(recorder.record),
		WithToolCallResultCallback(func(ToolStatus) { dedicated.Add(1) }),
	)
	if err != nil {
		t.Fatalf("ExecuteTools: %v", err)
	}
	events := recorder.matching(func(ev StreamEvent) bool { return ev.Type == StreamEventToolResult })
	if len(events) != 1 {
		t.Fatalf("tool result events = %d, want 1: %+v", len(events), events)
	}
	ev := events[0]
	if ev.ToolName != "echo" || ev.ToolCallID == "" || ev.ToolResult != "ok" || ev.ToolCallIndex != 0 || ev.AgentID != "" || ev.Error != nil {
		t.Fatalf("wrong outcome: %+v", ev)
	}
	if got := dedicated.Load(); got != 1 {
		t.Fatalf("dedicated callback calls = %d, want 1", got)
	}
}

func TestStreamToolResultEmitsOnlyFinalRetryOutcome(t *testing.T) {
	failure := errors.New("transient")
	runner := &streamTestRunner{results: []streamTestResult{{err: failure}, {result: "recovered"}}}
	tool := streamTestTool("retry", runner)
	var recorder eventRecorder

	_, err := ExecuteTools(newSequenceLLM(toolTurn("retry", `{}`)), NewEmptyFragment().AddMessage(UserMessageRole, "run"),
		WithTools(tool), DisableSinkState, WithIterations(1), WithMaxAttempts(2), WithStreamCallback(recorder.record))
	if err != nil {
		t.Fatalf("ExecuteTools: %v", err)
	}
	events := recorder.matching(func(ev StreamEvent) bool { return ev.Type == StreamEventToolResult })
	if len(events) != 1 || events[0].ToolResult != "recovered" || events[0].Error != nil {
		t.Fatalf("retry outcomes = %+v, want one successful final outcome", events)
	}
	if got := runner.callCount(); got != 2 {
		t.Fatalf("tool calls = %d, want 2", got)
	}
}

func TestStreamToolResultCarriesFinalRenderedFailureAndCause(t *testing.T) {
	failure := errors.New("permanent")
	runner := &streamTestRunner{results: []streamTestResult{{err: failure}, {err: failure}}}
	tool := streamTestTool("fail", runner)
	var recorder eventRecorder

	_, err := ExecuteTools(newSequenceLLM(toolTurn("fail", `{}`)), NewEmptyFragment().AddMessage(UserMessageRole, "run"),
		WithTools(tool), DisableSinkState, WithIterations(1), WithMaxAttempts(2), WithStreamCallback(recorder.record))
	if err != nil {
		t.Fatalf("ExecuteTools: %v", err)
	}
	events := recorder.matching(func(ev StreamEvent) bool { return ev.Type == StreamEventToolResult })
	if len(events) != 1 {
		t.Fatalf("failure events = %d, want 1: %+v", len(events), events)
	}
	if events[0].ToolResult != "Error running tool: permanent" || !errors.Is(events[0].Error, failure) {
		t.Fatalf("wrong failure outcome: %+v", events[0])
	}
}

func TestStreamParallelToolResultsKeepOriginalIndexes(t *testing.T) {
	ctx := streamTestContext(t)
	firstEntered := make(chan struct{})
	secondEntered := make(chan struct{})
	first := streamTestTool("first", &streamTestRunner{
		results: []streamTestResult{{result: "first-result"}}, started: firstEntered, waitFor: secondEntered, waitCtx: ctx,
	})
	second := streamTestTool("second", &streamTestRunner{
		results: []streamTestResult{{result: "second-result"}}, waitFor: firstEntered, started: secondEntered, waitCtx: ctx,
	})
	llm := newSequenceLLM(toolsTurn(toolCall{"first", `{}`}, toolCall{"second", `{}`}))
	var recorder eventRecorder

	f, err := ExecuteTools(llm, NewEmptyFragment().AddMessage(UserMessageRole, "run"),
		WithContext(ctx), WithTools(first, second), DisableSinkState, WithIterations(1), EnableParallelToolExecution, WithStreamCallback(recorder.record))
	if err != nil {
		t.Fatalf("ExecuteTools: %v", err)
	}
	events := recorder.matching(func(ev StreamEvent) bool { return ev.Type == StreamEventToolResult })
	if len(events) != 2 {
		t.Fatalf("parallel events = %d, want 2: %+v", len(events), events)
	}
	byID := make(map[string]StreamEvent, len(events))
	for _, ev := range events {
		byID[ev.ToolCallID] = ev
	}
	var callIDs []string
	for _, message := range f.Messages {
		for _, call := range message.ToolCalls {
			callIDs = append(callIDs, call.ID)
		}
	}
	if len(callIDs) != 2 {
		t.Fatalf("assistant tool calls = %v, want 2", callIDs)
	}
	if ev := byID[callIDs[0]]; ev.ToolName != "first" || ev.ToolResult != "first-result" || ev.ToolCallIndex != 0 {
		t.Fatalf("wrong first call correlation: %+v", ev)
	}
	if ev := byID[callIDs[1]]; ev.ToolName != "second" || ev.ToolResult != "second-result" || ev.ToolCallIndex != 1 {
		t.Fatalf("wrong second call correlation: %+v", ev)
	}
}

func TestStreamQuestionBatchReorderingKeepsOriginalIndexes(t *testing.T) {
	echo := streamTestTool("echo", &streamTestRunner{results: []streamTestResult{{result: "echo-result"}}})
	llm := newSequenceLLM(toolsTurn(toolCall{"echo", `{}`}, toolCall{UserQuestionToolName, `{"question":"continue?"}`}))
	var recorder eventRecorder

	_, err := ExecuteTools(llm, NewEmptyFragment().AddMessage(UserMessageRole, "run"),
		WithTools(echo), DisableSinkState, WithIterations(1), EnableParallelToolExecution,
		WithUserQuestions(func(context.Context, UserQuestion) (UserAnswer, error) {
			return UserAnswer{Text: "yes"}, nil
		}),
		WithStreamCallback(recorder.record),
	)
	if err != nil {
		t.Fatalf("ExecuteTools: %v", err)
	}
	events := recorder.matching(func(ev StreamEvent) bool { return ev.Type == StreamEventToolResult })
	if len(events) != 2 {
		t.Fatalf("executed outcomes = %+v, want ask_user then echo", events)
	}
	if events[0].ToolName != UserQuestionToolName || events[0].ToolCallIndex != 1 {
		t.Fatalf("question outcome lost original index: %+v", events[0])
	}
	if events[1].ToolName != "echo" || events[1].ToolCallIndex != 0 {
		t.Fatalf("echo outcome lost original index: %+v", events[1])
	}
}

func TestStreamToolResultSkipsUnexecutedQuestionSibling(t *testing.T) {
	echoRunner := &streamTestRunner{results: []streamTestResult{{result: "must not run"}}}
	echo := streamTestTool("echo", echoRunner)
	llm := newSequenceLLM(toolsTurn(toolCall{"echo", `{}`}, toolCall{UserQuestionToolName, `{"question":"continue?"}`}))
	var recorder eventRecorder

	_, err := ExecuteTools(llm, NewEmptyFragment().AddMessage(UserMessageRole, "run"),
		WithTools(echo), DisableSinkState, WithIterations(1), EnableParallelToolExecution,
		WithUserQuestions(func(context.Context, UserQuestion) (UserAnswer, error) {
			return UserAnswer{}, errors.New("unavailable")
		}),
		WithStreamCallback(recorder.record),
	)
	if err != nil {
		t.Fatalf("ExecuteTools: %v", err)
	}
	events := recorder.matching(func(ev StreamEvent) bool { return ev.Type == StreamEventToolResult })
	if len(events) != 1 || events[0].ToolName != UserQuestionToolName || events[0].ToolCallIndex != 1 {
		t.Fatalf("executed outcomes = %+v, want ask_user only", events)
	}
	if got := echoRunner.callCount(); got != 0 {
		t.Fatalf("skipped sibling executed %d times", got)
	}
}

func TestStreamToolResultNilCallbackPreservesExecution(t *testing.T) {
	runner := &streamTestRunner{results: []streamTestResult{{result: "ok"}}}
	tool := streamTestTool("echo", runner)
	f, err := ExecuteTools(newSequenceLLM(toolTurn("echo", `{}`)), NewEmptyFragment().AddMessage(UserMessageRole, "run"),
		WithTools(tool), DisableSinkState, WithIterations(1), WithStreamCallback(nil))
	if err != nil || len(f.Status.ToolResults) != 1 || f.Status.ToolResults[0].Result != "ok" {
		t.Fatalf("nil callback changed execution: result=%+v err=%v", f.Status.ToolResults, err)
	}
}

func TestStreamAgentCompletionEvents(t *testing.T) {
	tests := []struct {
		name       string
		background bool
		detach     bool
		failure    error
		wantStatus AgentStatusType
		wantResult string
	}{
		{name: "foreground success", wantStatus: AgentStatusCompleted, wantResult: "child result"},
		{name: "background success", background: true, wantStatus: AgentStatusCompleted, wantResult: "child result"},
		{name: "foreground failure", failure: errors.New("dispatch failed"), wantStatus: AgentStatusFailed},
		{name: "background failure", background: true, failure: errors.New("dispatch failed"), wantStatus: AgentStatusFailed},
		{name: "detached foreground success", detach: true, wantStatus: AgentStatusCompleted, wantResult: "child result"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := streamTestContext(t)
			manager := NewAgentManager()
			var recorder eventRecorder
			var managerInspected atomic.Bool
			spawned := make(chan *AgentState, 1)
			started := make(chan AgentRunSpec, 1)
			release := make(chan struct{})
			injected := make(chan openai.ChatCompletionMessage, 1)
			var completions atomic.Int32
			dispatcher := func(ctx context.Context, spec AgentRunSpec) (Fragment, error) {
				started <- spec
				// Executor progress is informational and must not become an
				// authoritative lifecycle event.
				if spec.Emit != nil {
					spec.Emit(AgentEvent{AgentID: spec.ID, Kind: "done", Result: "progress result"})
					spec.Emit(AgentEvent{AgentID: spec.ID, Kind: "error", Err: "progress error"})
				}
				if tt.detach {
					select {
					case <-release:
					case <-ctx.Done():
						return Fragment{}, ctx.Err()
					}
				}
				if tt.failure != nil {
					return Fragment{}, tt.failure
				}
				return NewFragment(openai.ChatCompletionMessage{Role: "assistant", Content: tt.wantResult}), nil
			}
			streamCB := func(ev StreamEvent) {
				if ev.AgentStatus != "" {
					_, ok := manager.Get(ev.AgentID)
					managerInspected.Store(ok)
				}
				recorder.record(ev)
			}
			runner := &spawnAgentRunner{
				llm: noToolMockLLM{}, manager: manager, ctx: ctx, dispatcher: dispatcher,
				streamCB: streamCB, messageInjectionChan: injected,
				agentSpawnCallback:      func(agent *AgentState) { spawned <- agent },
				agentCompletionCallback: func(*AgentState) { completions.Add(1) },
			}

			runDone := make(chan error, 1)
			go func() {
				_, _, err := runner.Run(SpawnAgentArgs{Task: "child", Background: tt.background})
				runDone <- err
			}()
			registeredAgent := awaitStreamTest(t, ctx, spawned, "agent registration")
			spec := awaitStreamTest(t, ctx, started, "dispatcher start")
			if registeredAgent.ID != spec.ID {
				t.Fatalf("spawned agent ID = %s, dispatcher ID = %s", registeredAgent.ID, spec.ID)
			}
			if tt.background || tt.detach {
				t.Cleanup(registeredAgent.Cancel)
			}
			if tt.detach {
				if err := manager.Detach(spec.ID); err != nil {
					t.Fatalf("Detach: %v", err)
				}
				close(release)
			}
			if err := awaitStreamTest(t, ctx, runDone, "spawn runner"); err != nil {
				t.Fatalf("Run: %v", err)
			}
			awaitStreamAgent(t, ctx, registeredAgent)
			agent := registeredAgent
			if agent.Status != tt.wantStatus {
				t.Fatalf("agent status = %q, want %q", agent.Status, tt.wantStatus)
			}
			events := recorder.matching(func(ev StreamEvent) bool { return ev.AgentStatus != "" })
			if len(events) != 1 {
				t.Fatalf("authoritative lifecycle events = %d, want 1: %+v", len(events), events)
			}
			ev := events[0]
			wantContent := tt.wantResult
			if tt.failure != nil {
				wantContent = "Failed: " + tt.failure.Error()
			}
			if ev.Type != StreamEventSubAgent || ev.AgentID != spec.ID || ev.AgentStatus != tt.wantStatus || ev.Content != wantContent {
				t.Fatalf("wrong completion event: %+v", ev)
			}
			if tt.failure != nil {
				if !errors.Is(ev.Error, tt.failure) || ev.FinishReason != "error" {
					t.Fatalf("wrong failed completion: %+v", ev)
				}
			} else if ev.Error != nil || ev.FinishReason != "stop" {
				t.Fatalf("wrong successful completion: %+v", ev)
			}
			if !managerInspected.Load() {
				t.Fatal("stream callback could not inspect manager")
			}
			if got := completions.Load(); got != 1 {
				t.Fatalf("existing completion callback calls = %d, want 1", got)
			}
			select {
			case <-injected:
			default:
				t.Fatal("existing completion injection did not fire")
			}
		})
	}
}

func TestStreamAgentCompletionHandlesEmptyDispatcherFragment(t *testing.T) {
	manager := NewAgentManager()
	var recorder eventRecorder
	runner := &spawnAgentRunner{
		llm: noToolMockLLM{}, manager: manager, ctx: context.Background(), streamCB: recorder.record,
		dispatcher: func(context.Context, AgentRunSpec) (Fragment, error) { return NewEmptyFragment(), nil },
	}
	if _, _, err := runner.Run(SpawnAgentArgs{Task: "empty", Background: false}); err != nil {
		t.Fatalf("Run: %v", err)
	}
	events := recorder.matching(func(ev StreamEvent) bool { return ev.AgentStatus != "" })
	if len(events) != 1 || events[0].AgentStatus != AgentStatusCompleted || events[0].Content != "" {
		t.Fatalf("empty result completion = %+v", events)
	}
}

func TestStreamChildToolResultsRetainTypeAndAgentID(t *testing.T) {
	for _, background := range []bool{false, true} {
		name := "foreground"
		if background {
			name = "background"
		}
		t.Run(name, func(t *testing.T) {
			ctx := streamTestContext(t)
			childTool := streamTestTool("echo", &streamTestRunner{results: []streamTestResult{{result: "child tool result"}}})
			manager := NewAgentManager()
			var recorder eventRecorder
			runner := &spawnAgentRunner{
				llm:         newSequenceLLM(toolTurn("echo", `{}`), replyTurn("child done")),
				parentTools: Tools{childTool}, manager: manager, ctx: ctx, streamCB: recorder.record,
			}
			_, idAny, err := runner.Run(SpawnAgentArgs{Task: "use echo", Background: background})
			if err != nil {
				t.Fatalf("Run: %v", err)
			}
			var id string
			if background {
				id, _ = idAny.(string)
				agent, ok := manager.Get(id)
				if !ok {
					t.Fatalf("agent %s was not registered", id)
				}
				t.Cleanup(agent.Cancel)
				awaitStreamAgent(t, ctx, agent)
			} else {
				agents := manager.List()
				if len(agents) != 1 {
					t.Fatalf("agents = %d, want 1", len(agents))
				}
				id = agents[0].ID
			}
			events := recorder.matching(func(ev StreamEvent) bool { return ev.Type == StreamEventToolResult })
			if len(events) != 1 || events[0].AgentID != id || events[0].ToolName != "echo" || events[0].ToolResult != "child tool result" {
				t.Fatalf("child tool outcomes = %+v", events)
			}
		})
	}
}

func TestStreamChildToolResultSurvivesFinishedAgentResume(t *testing.T) {
	childTool := streamTestTool("echo", &streamTestRunner{results: []streamTestResult{{result: "resumed tool result"}}})
	manager := NewAgentManager()
	var recorder eventRecorder
	llm := newSequenceLLM(toolTurn("echo", `{}`))
	o := defaultOptions()
	o.Apply(
		WithTools(childTool),
		EnableAgentSpawning,
		WithAgentManager(manager),
		WithStreamCallback(recorder.record),
	)
	prepared := prepareAgentTools(o, llm)
	resumeTool := Tools(prepared).Find("send_agent_message")
	if resumeTool == nil {
		t.Fatal("send_agent_message tool was not prepared")
	}
	// A stored AgentState does not currently retain its resolved tool allow-list.
	// Restore the fixture's child tool explicitly so this regression isolates
	// stream callback propagation through the real resume runner.
	definition, ok := resumeTool.(*ToolDefinition[SendAgentMessageArgs])
	if !ok {
		t.Fatalf("resume tool definition has type %T", resumeTool)
	}
	resumeRunner, ok := definition.ToolRunner.(*sendAgentMessageRunner)
	if !ok {
		t.Fatalf("resume runner has type %T", definition.ToolRunner)
	}
	resumeRunner.subOpts = append(resumeRunner.subOpts, WithTools(childTool))
	fragment := NewEmptyFragment().AddMessage(AssistantMessageRole, "previous result")
	agent := &AgentState{
		ID: "child-resume", Status: AgentStatusCompleted, Result: "previous result", Fragment: &fragment,
		done: make(chan struct{}),
	}
	close(agent.done)
	manager.Register(agent)

	_, _, err := resumeTool.Execute(map[string]any{"agent_id": agent.ID, "message": "continue"})
	if err != nil {
		t.Fatalf("resume tool: %v", err)
	}
	events := recorder.matching(func(ev StreamEvent) bool { return ev.Type == StreamEventToolResult })
	if len(events) != 1 || events[0].AgentID != agent.ID || events[0].ToolName != "echo" || events[0].ToolResult != "resumed tool result" {
		t.Fatalf("resumed child outcomes = %+v", events)
	}
}

func TestStreamToolResultSurvivesPlanningOptionConversion(t *testing.T) {
	tool := streamTestTool("echo", &streamTestRunner{results: []streamTestResult{{result: "converted result"}}})
	var recorder eventRecorder
	o := defaultOptions()
	o.Apply(WithTools(tool), WithStreamCallback(recorder.record), DisableSinkState, WithIterations(1))

	_, err := ExecuteTools(newSequenceLLM(toolTurn("echo", `{}`)), NewEmptyFragment().AddMessage(UserMessageRole, "run"),
		convertOptionsToFunctions(o)...)
	if err != nil {
		t.Fatalf("ExecuteTools: %v", err)
	}
	events := recorder.matching(func(ev StreamEvent) bool { return ev.Type == StreamEventToolResult })
	if len(events) != 1 || events[0].ToolResult != "converted result" {
		t.Fatalf("converted stream outcomes = %+v", events)
	}
}

func TestStreamAgentNilCallbackPreservesRun(t *testing.T) {
	runner := &spawnAgentRunner{
		llm: noToolMockLLM{}, manager: NewAgentManager(), ctx: context.Background(), streamCB: nil,
		dispatcher: func(context.Context, AgentRunSpec) (Fragment, error) {
			return NewFragment(openai.ChatCompletionMessage{Role: "assistant", Content: "ok"}), nil
		},
	}
	out, _, err := runner.Run(SpawnAgentArgs{Task: "nil callback", Background: false})
	if err != nil || out != "ok" {
		t.Fatalf("nil callback changed child run: out=%q err=%v", out, err)
	}
}
