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

type streamCorrelationLLM struct {
	streams [][]StreamEvent
	calls   int
}

func (m *streamCorrelationLLM) Ask(context.Context, Fragment) (Fragment, error) {
	return Fragment{}, errors.New("unexpected Ask fallback")
}

func (m *streamCorrelationLLM) CreateChatCompletion(context.Context, openai.ChatCompletionRequest) (LLMReply, LLMUsage, error) {
	return LLMReply{}, LLMUsage{}, errors.New("unexpected nonstream completion")
}

func (m *streamCorrelationLLM) CreateChatCompletionStream(context.Context, openai.ChatCompletionRequest) (<-chan StreamEvent, error) {
	if m.calls >= len(m.streams) {
		return nil, fmt.Errorf("unexpected stream completion %d", m.calls+1)
	}
	events := m.streams[m.calls]
	m.calls++
	ch := make(chan StreamEvent, len(events))
	for _, event := range events {
		ch <- event
	}
	close(ch)
	return ch, nil
}

type nonStreamCorrelationLLM struct {
	id    string
	calls int
	asks  int
}

func (m *nonStreamCorrelationLLM) Ask(_ context.Context, f Fragment) (Fragment, error) {
	m.asks++
	return f.AddMessage(AssistantMessageRole, "done"), nil
}

func (m *nonStreamCorrelationLLM) CreateChatCompletion(context.Context, openai.ChatCompletionRequest) (LLMReply, LLMUsage, error) {
	m.calls++
	return LLMReply{ChatCompletionResponse: openai.ChatCompletionResponse{
		Choices: []openai.ChatCompletionChoice{{Message: openai.ChatCompletionMessage{
			Role: AssistantMessageRole.String(),
			ToolCalls: []openai.ToolCall{{
				ID:       m.id,
				Type:     openai.ToolTypeFunction,
				Function: openai.FunctionCall{Name: "echo", Arguments: `{"marker":"nonstream"}`},
			}},
		}}},
	}}, LLMUsage{}, nil
}

type streamMarkerRunner struct{}

func (streamMarkerRunner) Run(args map[string]any) (string, any, error) {
	return fmt.Sprint(args["marker"]), nil, nil
}

func streamMarkerTool() ToolDefinitionInterface {
	return NewToolDefinition[map[string]any](streamMarkerRunner{}, map[string]any{
		"type": "object",
		"properties": map[string]any{
			"marker": map[string]any{"type": "string"},
		},
	}, "echo", "echo")
}

func streamCorrelationTurn(id string, index int, name, args string) []StreamEvent {
	return []StreamEvent{
		{Type: StreamEventToolCall, ToolCallID: id, ToolCallIndex: index, ToolName: name, ToolArgs: args},
		{Type: StreamEventDone, FinishReason: "tool_calls"},
	}
}

func streamFinalTurn() []StreamEvent {
	return []StreamEvent{
		{Type: StreamEventContent, Content: "done"},
		{Type: StreamEventDone, FinishReason: "stop"},
	}
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

func TestStreamingToolResultsPreserveOutOfOrderProviderCorrelation(t *testing.T) {
	tool := streamMarkerTool()
	llm := &streamCorrelationLLM{streams: [][]StreamEvent{{
		{Type: StreamEventToolCall, ToolCallID: "provider-index-1", ToolCallIndex: 1, ToolName: "echo", ToolArgs: `{"marker":"provider-`},
		{Type: StreamEventToolCall, ToolCallIndex: 1, ToolArgs: `one"}`},
		{Type: StreamEventToolCall, ToolCallID: "provider-index-0", ToolCallIndex: 0, ToolName: "echo", ToolArgs: `{"marker":"provider-zero"}`},
		{Type: StreamEventDone, FinishReason: "tool_calls"},
	}, streamFinalTurn()}}
	var recorder eventRecorder

	f, err := ExecuteTools(llm, NewEmptyFragment().AddMessage(UserMessageRole, "run"),
		WithTools(tool), DisableSinkState, WithIterations(1), WithStreamCallback(recorder.record))
	if err != nil {
		t.Fatalf("ExecuteTools: %v", err)
	}
	if llm.calls != 2 {
		t.Fatalf("stream completions = %d, want selection + final answer", llm.calls)
	}

	results := recorder.matching(func(ev StreamEvent) bool { return ev.Type == StreamEventToolResult })
	if len(results) != 2 {
		t.Fatalf("tool result events = %d, want 2: %+v", len(results), results)
	}
	wantIDs := []string{"provider-index-0", "provider-index-1"}
	wantResults := []string{"provider-zero", "provider-one"}
	for i := range results {
		if results[i].ToolCallID != wantIDs[i] || results[i].ToolCallIndex != i || results[i].ToolResult != wantResults[i] {
			t.Fatalf("result event %d = %+v, want id=%q index=%d result=%q", i, results[i], wantIDs[i], i, wantResults[i])
		}
	}

	if len(f.Messages) < 4 {
		t.Fatalf("transcript messages = %d, want at least user + assistant + 2 tools: %+v", len(f.Messages), f.Messages)
	}
	assistant := f.Messages[1]
	if len(assistant.ToolCalls) != 2 {
		t.Fatalf("assistant tool calls = %d, want 2: %+v", len(assistant.ToolCalls), assistant.ToolCalls)
	}
	for i := range assistant.ToolCalls {
		if assistant.ToolCalls[i].ID != wantIDs[i] {
			t.Fatalf("assistant tool call %d id = %q, want %q", i, assistant.ToolCalls[i].ID, wantIDs[i])
		}
		if f.Messages[i+2].Role != "tool" || f.Messages[i+2].ToolCallID != wantIDs[i] || f.Messages[i+2].Content != wantResults[i] {
			t.Fatalf("tool transcript message %d = %+v, want id=%q result=%q", i, f.Messages[i+2], wantIDs[i], wantResults[i])
		}
	}
}

func TestStreamingToolResultGeneratesOnlyMissingIDAndKeepsSparseIndex(t *testing.T) {
	tool := streamMarkerTool()
	llm := &streamCorrelationLLM{streams: [][]StreamEvent{
		streamCorrelationTurn("", 4, "echo", `{"marker":"missing-id"}`),
		streamFinalTurn(),
	}}
	var recorder eventRecorder

	f, err := ExecuteTools(llm, NewEmptyFragment().AddMessage(UserMessageRole, "run"),
		WithTools(tool), DisableSinkState, WithIterations(1), WithStreamCallback(recorder.record))
	if err != nil {
		t.Fatalf("ExecuteTools: %v", err)
	}
	results := recorder.matching(func(ev StreamEvent) bool { return ev.Type == StreamEventToolResult })
	if len(results) != 1 {
		t.Fatalf("tool result events = %d, want 1: %+v", len(results), results)
	}
	generatedID := results[0].ToolCallID
	if generatedID == "" || results[0].ToolCallIndex != 4 {
		t.Fatalf("generated correlation = %+v, want non-empty id and index 4", results[0])
	}
	if f.Messages[1].ToolCalls[0].ID != generatedID || f.Messages[2].ToolCallID != generatedID || f.Status.ToolResults[0].ToolArguments.ID != generatedID {
		t.Fatalf("generated id not shared by assistant, tool, status, and event: id=%q messages=%+v status=%+v", generatedID, f.Messages, f.Status.ToolResults)
	}
}

func TestNonStreamingSelectionPreservesProviderIDWithNilCallback(t *testing.T) {
	tool := streamMarkerTool()
	llm := &nonStreamCorrelationLLM{id: "provider-nonstream"}

	f, err := ExecuteTools(llm, NewEmptyFragment().AddMessage(UserMessageRole, "run"),
		WithTools(tool), DisableSinkState, WithIterations(1), WithStreamCallback(nil))
	if err != nil {
		t.Fatalf("ExecuteTools: %v", err)
	}
	if llm.calls != 1 || llm.asks != 1 {
		t.Fatalf("nonstream calls = completions:%d asks:%d, want 1 each", llm.calls, llm.asks)
	}
	if f.Messages[1].ToolCalls[0].ID != "provider-nonstream" || f.Messages[2].ToolCallID != "provider-nonstream" || f.Status.ToolResults[0].ToolArguments.ID != "provider-nonstream" {
		t.Fatalf("provider id not preserved through nil-callback nonstream execution: messages=%+v status=%+v", f.Messages, f.Status.ToolResults)
	}
}

func TestForcedReasoningResultUsesExecutedParameterCallCorrelation(t *testing.T) {
	tool := streamMarkerTool()
	llm := &streamCorrelationLLM{streams: [][]StreamEvent{
		streamCorrelationTurn("reasoning-selection", 0, "reasoning", `{"reasoning":"use echo"}`),
		streamCorrelationTurn("intention-selection", 0, "pick_tool", `{"tool":"echo","reasoning":"use echo"}`),
		streamCorrelationTurn("reasoning-parameters", 0, "reasoning", `{"reasoning":"set marker"}`),
		streamCorrelationTurn("provider-parameters", 6, "echo", `{"marker":"enhanced"}`),
		streamFinalTurn(),
	}}
	var recorder eventRecorder

	f, err := ExecuteTools(llm, NewEmptyFragment().AddMessage(UserMessageRole, "run"),
		WithTools(tool), WithForceReasoning(), WithIterations(1), WithStreamCallback(recorder.record))
	if err != nil {
		t.Fatalf("ExecuteTools: %v", err)
	}
	if llm.calls != 5 {
		t.Fatalf("stream completions = %d, want 4 selection/parameter calls + final answer", llm.calls)
	}
	results := recorder.matching(func(ev StreamEvent) bool { return ev.Type == StreamEventToolResult })
	if len(results) != 1 || results[0].ToolCallID != "provider-parameters" || results[0].ToolCallIndex != 6 || results[0].ToolResult != "enhanced" {
		t.Fatalf("forced reasoning result correlation = %+v, want executed parameter call id/index", results)
	}
	if f.Messages[1].ToolCalls[0].ID != "provider-parameters" || f.Messages[2].ToolCallID != "provider-parameters" {
		t.Fatalf("forced reasoning transcript lost executed parameter call id: %+v", f.Messages)
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
