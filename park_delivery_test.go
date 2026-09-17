package cogito

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/sashabaranov/go-openai"
)

func TestWithOnParkSnapshotOwnsCompleteMessages(t *testing.T) {
	injected := make(chan openai.ChatCompletionMessage, 1)
	var pending atomic.Bool
	pending.Store(true)
	parked := make(chan struct{})
	var once sync.Once
	initial := NewFragment(openai.ChatCompletionMessage{
		Role:    "user",
		Content: "original",
		MultiContent: []openai.ChatMessagePart{{
			Type:     openai.ChatMessagePartTypeImageURL,
			ImageURL: &openai.ChatMessageImageURL{URL: "https://example.invalid/image.png"},
		}},
		FunctionCall: &openai.FunctionCall{Name: "original-function"},
		ToolCalls: []openai.ToolCall{{
			ID:       "original-call",
			Function: openai.FunctionCall{Name: "original-tool"},
		}},
	})

	done := make(chan struct{})
	var result Fragment
	var runErr error
	go func() {
		result, runErr = ExecuteTools(noToolMockLLM{}, initial,
			DisableSinkState,
			WithMessageInjectionChan(injected),
			WithPendingWork(func() bool { return pending.Load() }),
			WithOnParkSnapshot(func(snapshot Fragment, reply string) {
				if reply != "sub-agent done" {
					t.Errorf("park reply = %q, want sub-agent done", reply)
				}
				if got := snapshot.Messages[len(snapshot.Messages)-1].Content; got != reply {
					t.Errorf("snapshot last message = %q, want parked reply %q", got, reply)
				}
				snapshot.Messages[0].Content = "mutated"
				snapshot.Messages[0].MultiContent[0].ImageURL.URL = "mutated"
				snapshot.Messages[0].FunctionCall.Name = "mutated"
				snapshot.Messages[0].ToolCalls[0].ID = "mutated"
				once.Do(func() { close(parked) })
			}),
			WithIterations(5),
		)
		close(done)
	}()

	select {
	case <-parked:
	case <-time.After(3 * time.Second):
		t.Fatal("loop did not park")
	}
	pending.Store(false)
	injected <- openai.ChatCompletionMessage{Role: "user", Content: "wake"}
	select {
	case <-done:
	case <-time.After(3 * time.Second):
		t.Fatal("loop did not finish after wake")
	}
	if runErr != nil {
		t.Fatalf("ExecuteTools: %v", runErr)
	}
	got := result.Messages[0]
	if got.Content != "original" || got.MultiContent[0].ImageURL.URL != "https://example.invalid/image.png" || got.FunctionCall.Name != "original-function" || got.ToolCalls[0].ID != "original-call" {
		t.Fatalf("snapshot mutation escaped into live fragment: %#v", got)
	}
}

func TestWithOnParkSnapshotFiresAtSinkGate(t *testing.T) {
	injected := make(chan openai.ChatCompletionMessage, 1)
	var pending atomic.Bool
	pending.Store(true)
	parked := make(chan struct{})
	var once sync.Once
	llm := newSequenceLLM(toolTurn("reply", `{}`), replyTurn("done"))
	done := make(chan struct{})
	go func() {
		_, _ = ExecuteTools(llm, NewEmptyFragment().AddMessage(UserMessageRole, "hello"),
			WithMessageInjectionChan(injected),
			WithPendingWork(func() bool { return pending.Load() }),
			WithOnParkSnapshot(func(snapshot Fragment, reply string) {
				if reply != "" {
					t.Errorf("sink park reply = %q, want empty", reply)
				}
				if len(snapshot.Messages) != 1 || snapshot.Messages[0].Content != "hello" {
					t.Errorf("sink snapshot messages = %#v", snapshot.Messages)
				}
				once.Do(func() { close(parked) })
			}),
			WithIterations(5),
		)
		close(done)
	}()

	select {
	case <-parked:
	case <-time.After(3 * time.Second):
		t.Fatal("sink loop did not park")
	}
	pending.Store(false)
	injected <- openai.ChatCompletionMessage{Role: "user", Content: "wake"}
	select {
	case <-done:
	case <-time.After(3 * time.Second):
		t.Fatal("sink loop did not finish after wake")
	}
}

func TestBackgroundCompletionWaitsForFullInbox(t *testing.T) {
	injected := make(chan openai.ChatCompletionMessage, 1)
	injected <- openai.ChatCompletionMessage{Role: "user", Content: "already queued"}
	manager := NewAgentManager()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	callbackEntered := make(chan struct{})
	runner := &spawnAgentRunner{
		manager:              manager,
		ctx:                  ctx,
		messageInjectionChan: injected,
		agentCompletionCallback: func(*AgentState) {
			close(callbackEntered)
		},
		dispatcher: func(context.Context, AgentRunSpec) (Fragment, error) {
			return NewEmptyFragment().AddMessage(AssistantMessageRole, "child result"), nil
		},
	}

	_, idValue, err := runner.Run(SpawnAgentArgs{Task: "background task", Background: true})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	id := idValue.(string)
	select {
	case <-callbackEntered:
	case <-time.After(3 * time.Second):
		t.Fatal("completion callback did not run")
	}
	if manager.HasRunning() {
		t.Fatal("agent should be terminal before completion callback")
	}
	if !manager.HasPendingWork() {
		t.Fatal("completion publication should remain pending while inbox is full")
	}
	if got := (<-injected).Content; got != "already queued" {
		t.Fatalf("first queued message = %q", got)
	}
	select {
	case msg := <-injected:
		if !strings.Contains(msg.Content, "child result") {
			t.Fatalf("completion message = %q", msg.Content)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("completion was dropped when inbox was full")
	}
	if _, err := manager.Wait(id); err != nil {
		t.Fatalf("Wait: %v", err)
	}
}

func TestBackgroundCompletionCancelsWhileInboxFull(t *testing.T) {
	injected := make(chan openai.ChatCompletionMessage, 1)
	injected <- openai.ChatCompletionMessage{Role: "user", Content: "already queued"}
	manager := NewAgentManager()
	ctx, cancel := context.WithCancel(context.Background())
	callbackEntered := make(chan struct{})
	runner := &spawnAgentRunner{
		manager:              manager,
		ctx:                  ctx,
		messageInjectionChan: injected,
		agentCompletionCallback: func(*AgentState) {
			close(callbackEntered)
		},
		dispatcher: func(context.Context, AgentRunSpec) (Fragment, error) {
			return NewEmptyFragment().AddMessage(AssistantMessageRole, "child result"), nil
		},
	}
	_, idValue, err := runner.Run(SpawnAgentArgs{Task: "background task", Background: true})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	id := idValue.(string)
	select {
	case <-callbackEntered:
	case <-time.After(3 * time.Second):
		t.Fatal("completion callback did not run")
	}
	if !manager.HasPendingWork() {
		t.Fatal("completion publication should remain pending while inbox is full")
	}
	cancel()
	waited := make(chan struct{})
	go func() {
		_, _ = manager.Wait(id)
		close(waited)
	}()
	select {
	case <-waited:
	case <-time.After(3 * time.Second):
		t.Fatal("completion publisher ignored parent cancellation")
	}
}

func TestBackgroundCompletionStaysPendingDuringCallback(t *testing.T) {
	injected := make(chan openai.ChatCompletionMessage, 1)
	manager := NewAgentManager()
	callbackEntered := make(chan struct{})
	releaseCallback := make(chan struct{})
	runner := &spawnAgentRunner{
		manager:              manager,
		ctx:                  context.Background(),
		messageInjectionChan: injected,
		agentCompletionCallback: func(*AgentState) {
			close(callbackEntered)
			<-releaseCallback
		},
		dispatcher: func(context.Context, AgentRunSpec) (Fragment, error) {
			return NewEmptyFragment().AddMessage(AssistantMessageRole, "child result"), nil
		},
	}
	if _, _, err := runner.Run(SpawnAgentArgs{Task: "background task", Background: true}); err != nil {
		t.Fatalf("Run: %v", err)
	}
	select {
	case <-callbackEntered:
	case <-time.After(3 * time.Second):
		t.Fatal("completion callback did not run")
	}
	if manager.HasRunning() {
		t.Fatal("agent should be terminal while completion callback runs")
	}
	if !manager.HasPendingWork() {
		t.Fatal("callback/publication gap was reported idle")
	}
	parked := make(chan struct{})
	parentDone := make(chan error, 1)
	go func() {
		_, err := ExecuteTools(noToolMockLLM{}, NewEmptyFragment().AddMessage(UserMessageRole, "start"),
			DisableSinkState,
			WithAgentManager(manager),
			WithMessageInjectionChan(injected),
			WithOnPark(func(string) { close(parked) }),
			WithIterations(5),
		)
		parentDone <- err
	}()
	select {
	case <-parked:
	case <-time.After(3 * time.Second):
		t.Fatal("parent loop did not park during callback/publication gap")
	}
	close(releaseCallback)
	select {
	case err := <-parentDone:
		if err != nil {
			t.Fatalf("parent ExecuteTools: %v", err)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("parent loop did not consume completion after callback")
	}
}

func TestForegroundCompletionDoesNotPublishToParentInbox(t *testing.T) {
	injected := make(chan openai.ChatCompletionMessage, 1)
	injected <- openai.ChatCompletionMessage{Role: "user", Content: "already queued"}
	runner := &spawnAgentRunner{
		manager:              NewAgentManager(),
		ctx:                  context.Background(),
		messageInjectionChan: injected,
		dispatcher: func(context.Context, AgentRunSpec) (Fragment, error) {
			return NewEmptyFragment().AddMessage(AssistantMessageRole, "child result"), nil
		},
	}
	done := make(chan error, 1)
	go func() {
		_, _, err := runner.Run(SpawnAgentArgs{Task: "foreground task", Background: false})
		done <- err
	}()
	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("Run: %v", err)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("foreground completion blocked on parent inbox")
	}
	if got := len(injected); got != 1 {
		t.Fatalf("foreground completion changed parent inbox length to %d", got)
	}
}

type blockedFirstReplyLLM struct {
	started chan struct{}
	release chan struct{}
	mu      sync.Mutex
	calls   int
	seen    []openai.ChatCompletionRequest
}

func (m *blockedFirstReplyLLM) Ask(_ context.Context, f Fragment) (Fragment, error) {
	return f.AddMessage(AssistantMessageRole, "final"), nil
}

func (m *blockedFirstReplyLLM) CreateChatCompletion(_ context.Context, req openai.ChatCompletionRequest) (LLMReply, LLMUsage, error) {
	m.mu.Lock()
	m.calls++
	call := m.calls
	m.seen = append(m.seen, req)
	m.mu.Unlock()
	if call == 1 {
		close(m.started)
		<-m.release
	}
	content := "first reply"
	if call > 1 {
		content = "after completion"
	}
	return LLMReply{ChatCompletionResponse: openai.ChatCompletionResponse{Choices: []openai.ChatCompletionChoice{{
		Message: openai.ChatCompletionMessage{Role: "assistant", Content: content},
	}}}}, LLMUsage{}, nil
}

func TestExecuteToolsConsumesCompletionQueuedDuringModelCall(t *testing.T) {
	injected := make(chan openai.ChatCompletionMessage, 1)
	manager := NewAgentManager()
	llm := &blockedFirstReplyLLM{started: make(chan struct{}), release: make(chan struct{})}
	done := make(chan struct{})
	var result Fragment
	var runErr error
	go func() {
		result, runErr = ExecuteTools(llm, NewEmptyFragment().AddMessage(UserMessageRole, "start"),
			DisableSinkState,
			WithAgentManager(manager),
			WithMessageInjectionChan(injected),
			WithIterations(5),
		)
		close(done)
	}()
	<-llm.started
	runner := &spawnAgentRunner{
		manager:              manager,
		ctx:                  context.Background(),
		messageInjectionChan: injected,
		dispatcher: func(context.Context, AgentRunSpec) (Fragment, error) {
			return NewEmptyFragment().AddMessage(AssistantMessageRole, "child result"), nil
		},
	}
	if _, _, err := runner.Run(SpawnAgentArgs{Task: "background task", Background: true}); err != nil {
		t.Fatalf("Run: %v", err)
	}
	deadline := time.After(3 * time.Second)
	for len(injected) == 0 {
		select {
		case <-deadline:
			t.Fatal("completion was not queued during model call")
		default:
			time.Sleep(time.Millisecond)
		}
	}
	close(llm.release)
	select {
	case <-done:
	case <-time.After(3 * time.Second):
		t.Fatal("parent loop did not finish")
	}
	if runErr != nil {
		t.Fatalf("ExecuteTools: %v", runErr)
	}
	if llm.calls < 2 {
		t.Fatalf("model calls = %d, queued completion was not consumed", llm.calls)
	}
	found := false
	for _, msg := range result.Messages {
		if strings.Contains(msg.Content, "child result") {
			found = true
		}
	}
	if !found {
		t.Fatalf("result omitted queued completion: %#v", result.Messages)
	}
}

type blockedFirstSinkLLM struct {
	started chan struct{}
	release chan struct{}
	mu      sync.Mutex
	calls   int
}

func (m *blockedFirstSinkLLM) Ask(_ context.Context, f Fragment) (Fragment, error) {
	return f.AddMessage(AssistantMessageRole, "final"), nil
}

func (m *blockedFirstSinkLLM) CreateChatCompletion(_ context.Context, _ openai.ChatCompletionRequest) (LLMReply, LLMUsage, error) {
	m.mu.Lock()
	m.calls++
	call := m.calls
	m.mu.Unlock()
	message := openai.ChatCompletionMessage{Role: "assistant", Content: "after completion"}
	if call == 1 {
		close(m.started)
		<-m.release
		message = openai.ChatCompletionMessage{
			Role: "assistant",
			ToolCalls: []openai.ToolCall{{
				ID:       "reply-call",
				Type:     openai.ToolTypeFunction,
				Function: openai.FunctionCall{Name: "reply", Arguments: `{}`},
			}},
		}
	}
	return LLMReply{ChatCompletionResponse: openai.ChatCompletionResponse{Choices: []openai.ChatCompletionChoice{{Message: message}}}}, LLMUsage{}, nil
}

func TestExecuteToolsConsumesCompletionQueuedDuringSinkDecision(t *testing.T) {
	injected := make(chan openai.ChatCompletionMessage, 1)
	manager := NewAgentManager()
	llm := &blockedFirstSinkLLM{started: make(chan struct{}), release: make(chan struct{})}
	done := make(chan struct{})
	var result Fragment
	var runErr error
	go func() {
		result, runErr = ExecuteTools(llm, NewEmptyFragment().AddMessage(UserMessageRole, "start"),
			WithAgentManager(manager),
			WithMessageInjectionChan(injected),
			WithIterations(5),
		)
		close(done)
	}()
	<-llm.started
	runner := &spawnAgentRunner{
		manager:              manager,
		ctx:                  context.Background(),
		messageInjectionChan: injected,
		dispatcher: func(context.Context, AgentRunSpec) (Fragment, error) {
			return NewEmptyFragment().AddMessage(AssistantMessageRole, "child sink result"), nil
		},
	}
	if _, _, err := runner.Run(SpawnAgentArgs{Task: "background task", Background: true}); err != nil {
		t.Fatalf("Run: %v", err)
	}
	deadline := time.After(3 * time.Second)
	for len(injected) == 0 {
		select {
		case <-deadline:
			t.Fatal("completion was not queued during sink decision")
		default:
			time.Sleep(time.Millisecond)
		}
	}
	close(llm.release)
	select {
	case <-done:
	case <-time.After(3 * time.Second):
		t.Fatal("parent loop did not finish")
	}
	if runErr != nil {
		t.Fatalf("ExecuteTools: %v", runErr)
	}
	if llm.calls < 2 {
		t.Fatalf("model calls = %d, queued sink completion was not consumed", llm.calls)
	}
	found := false
	for _, msg := range result.Messages {
		if strings.Contains(msg.Content, "child sink result") {
			found = true
		}
	}
	if !found {
		t.Fatalf("result omitted queued sink completion: %#v", result.Messages)
	}
}

func TestManagerPendingWorkPrecedesEmbedderPredicate(t *testing.T) {
	injected := make(chan openai.ChatCompletionMessage, 1)
	manager := NewAgentManager()
	manager.Register(&AgentState{ID: "running", Status: AgentStatusRunning})
	var calls atomic.Int64
	parked := make(chan struct{})
	done := make(chan struct{})
	go func() {
		_, _ = ExecuteTools(noToolMockLLM{}, NewEmptyFragment().AddMessage(UserMessageRole, "start"),
			DisableSinkState,
			WithAgentManager(manager),
			WithMessageInjectionChan(injected),
			WithPendingWork(func() bool { calls.Add(1); return false }),
			WithOnPark(func(string) { close(parked) }),
			WithIterations(5),
		)
		close(done)
	}()
	select {
	case <-parked:
	case <-time.After(3 * time.Second):
		t.Fatal("loop did not park for running child")
	}
	if got := calls.Load(); got != 0 {
		t.Fatalf("embedder predicate called %d times while manager had pending work", got)
	}
	manager.mu.Lock()
	manager.agents["running"].Status = AgentStatusCompleted
	manager.mu.Unlock()
	injected <- openai.ChatCompletionMessage{Role: "user", Content: "wake"}
	select {
	case <-done:
	case <-time.After(3 * time.Second):
		t.Fatal("loop did not finish")
	}
	if calls.Load() == 0 {
		t.Fatal("embedder predicate was never evaluated after manager became idle")
	}
}

func TestExecuteToolsConsumesInjectionPublishedByPendingPredicate(t *testing.T) {
	injected := make(chan openai.ChatCompletionMessage, 1)
	var calls atomic.Int64
	result, err := ExecuteTools(noToolMockLLM{}, NewEmptyFragment().AddMessage(UserMessageRole, "start"),
		DisableSinkState,
		WithMessageInjectionChan(injected),
		WithPendingWork(func() bool {
			if calls.Add(1) == 1 {
				injected <- openai.ChatCompletionMessage{Role: "user", Content: "published at exit gate"}
			}
			return false
		}),
		WithIterations(5),
	)
	if err != nil {
		t.Fatalf("ExecuteTools: %v", err)
	}
	for _, message := range result.Messages {
		if message.Content == "published at exit gate" {
			return
		}
	}
	t.Fatalf("result omitted injection published during exit admission: %#v", result.Messages)
}

type finalResponseCaptureLLM struct {
	mu        sync.Mutex
	turn      scriptedTurn
	callback  *atomic.Int64
	askInput  Fragment
	askCalled int
}

func (m *finalResponseCaptureLLM) Ask(_ context.Context, f Fragment) (Fragment, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.askCalled++
	m.askInput = f
	if m.callback != nil && m.callback.Load() != 1 {
		return Fragment{}, errors.New("final Ask started before callback")
	}
	return f.AddMessage(AssistantMessageRole, "final response"), nil
}

func (m *finalResponseCaptureLLM) CreateChatCompletion(_ context.Context, _ openai.ChatCompletionRequest) (LLMReply, LLMUsage, error) {
	m.mu.Lock()
	turn := m.turn
	m.turn = replyTurn("done")
	m.mu.Unlock()
	message := openai.ChatCompletionMessage{Role: AssistantMessageRole.String(), Content: turn.content}
	for i, call := range turn.calls {
		message.ToolCalls = append(message.ToolCalls, openai.ToolCall{
			ID:       fmt.Sprintf("final-call-%d", i),
			Type:     openai.ToolTypeFunction,
			Function: openai.FunctionCall{Name: call.name, Arguments: call.args},
		})
	}
	return LLMReply{ChatCompletionResponse: openai.ChatCompletionResponse{Choices: []openai.ChatCompletionChoice{{Message: message}}}}, LLMUsage{}, nil
}

type finalEchoRunner struct{}

func (finalEchoRunner) Run(_ struct{}) (string, any, error) { return "echoed", nil, nil }

func finalEchoTool() ToolDefinitionInterface {
	return NewToolDefinition(finalEchoRunner{}, struct{}{}, "echo", "echo")
}

func assertFinalAskContains(t *testing.T, llm *finalResponseCaptureLLM, content string) {
	t.Helper()
	llm.mu.Lock()
	defer llm.mu.Unlock()
	if llm.askCalled != 1 {
		t.Fatalf("final Ask calls = %d, want 1", llm.askCalled)
	}
	for _, message := range llm.askInput.Messages {
		if message.Content == content {
			return
		}
	}
	t.Fatalf("final Ask omitted %q: %#v", content, llm.askInput.Messages)
}

func TestOnBeforeFinalResponseDrainsMaxIterationInjection(t *testing.T) {
	injected := make(chan openai.ChatCompletionMessage, 1)
	var callbacks atomic.Int64
	llm := &finalResponseCaptureLLM{
		turn:     toolTurn("echo", `{}`),
		callback: &callbacks,
	}
	_, err := ExecuteTools(llm, NewEmptyFragment().AddMessage(UserMessageRole, "start"),
		WithTools(finalEchoTool()),
		WithIterations(1),
		WithMessageInjectionChan(injected),
		WithOnBeforeFinalResponse(func() {
			callbacks.Add(1)
			injected <- openai.ChatCompletionMessage{Role: "user", Content: "accepted before limit"}
		}),
	)
	if err != nil {
		t.Fatalf("ExecuteTools: %v", err)
	}
	if got := callbacks.Load(); got != 1 {
		t.Fatalf("callbacks = %d, want 1", got)
	}
	assertFinalAskContains(t, llm, "accepted before limit")
}

func TestOnBeforeFinalResponseDrainsSinkInjection(t *testing.T) {
	injected := make(chan openai.ChatCompletionMessage, 1)
	var callbacks atomic.Int64
	llm := &finalResponseCaptureLLM{
		turn:     toolTurn("reply", `{}`),
		callback: &callbacks,
	}
	_, err := ExecuteTools(llm, NewEmptyFragment().AddMessage(UserMessageRole, "start"),
		WithIterations(3),
		WithMessageInjectionChan(injected),
		WithOnBeforeFinalResponse(func() {
			callbacks.Add(1)
			injected <- openai.ChatCompletionMessage{Role: "user", Content: "accepted before sink"}
		}),
	)
	if !errors.Is(err, ErrNoToolSelected) {
		t.Fatalf("ExecuteTools error = %v, want ErrNoToolSelected", err)
	}
	if got := callbacks.Load(); got != 1 {
		t.Fatalf("callbacks = %d, want 1", got)
	}
	assertFinalAskContains(t, llm, "accepted before sink")
}

func TestOnBeforeFinalResponseDoesNotPropagateToSpawnedAgents(t *testing.T) {
	var callbacks atomic.Int64
	o := defaultOptions()
	o.Apply(
		EnableAgentSpawning,
		WithAgentLLM(noToolMockLLM{}),
		WithOnBeforeFinalResponse(func() { callbacks.Add(1) }),
	)
	prepared := prepareAgentTools(o, noToolMockLLM{})
	if len(prepared) == 0 {
		t.Fatal("agent tools were not prepared")
	}
	spawnTool, ok := prepared[0].(*ToolDefinition[SpawnAgentArgs])
	if !ok {
		t.Fatalf("spawn tool type = %T", prepared[0])
	}
	runner, ok := spawnTool.ToolRunner.(*spawnAgentRunner)
	if !ok {
		t.Fatalf("spawn runner type = %T", spawnTool.ToolRunner)
	}
	child := defaultOptions()
	child.Apply(runner.parentOpts...)
	if child.onBeforeFinalResponse != nil {
		t.Fatal("parent final-response callback propagated to spawned agent")
	}
}
