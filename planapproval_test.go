package cogito

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/mudler/cogito/structures"
	"github.com/sashabaranov/go-openai"
)

type approvalScriptEntry struct {
	name string
	args string
	err  error
}

type approvalLLM struct {
	mu       sync.Mutex
	script   []approvalScriptEntry
	next     int
	asks     []Fragment
	requests []openai.ChatCompletionRequest
}

func newApprovalLLM(script ...approvalScriptEntry) *approvalLLM {
	return &approvalLLM{script: script}
}

func (m *approvalLLM) Ask(ctx context.Context, f Fragment) (Fragment, error) {
	if err := ctx.Err(); err != nil {
		return f, err
	}
	m.mu.Lock()
	m.asks = append(m.asks, f)
	m.mu.Unlock()
	return f.AddMessage(AssistantMessageRole, "draft proposal"), nil
}

func (m *approvalLLM) CreateChatCompletion(ctx context.Context, request openai.ChatCompletionRequest) (LLMReply, LLMUsage, error) {
	if err := ctx.Err(); err != nil {
		return LLMReply{}, LLMUsage{}, err
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	m.requests = append(m.requests, request)
	if m.next >= len(m.script) {
		return LLMReply{}, LLMUsage{}, fmt.Errorf("approval script exhausted after %d requests", m.next)
	}
	entry := m.script[m.next]
	m.next++
	if entry.err != nil {
		return LLMReply{}, LLMUsage{}, entry.err
	}
	response := openai.ChatCompletionResponse{Choices: []openai.ChatCompletionChoice{{
		Message: openai.ChatCompletionMessage{
			Role: AssistantMessageRole.String(),
			ToolCalls: []openai.ToolCall{{
				ID:   fmt.Sprintf("approval-call-%d", m.next),
				Type: openai.ToolTypeFunction,
				Function: openai.FunctionCall{
					Name:      entry.name,
					Arguments: entry.args,
				},
			}},
		},
	}}}
	return LLMReply{ChatCompletionResponse: response}, LLMUsage{}, nil
}

func (m *approvalLLM) snapshot() ([]Fragment, []openai.ChatCompletionRequest, int, int) {
	m.mu.Lock()
	defer m.mu.Unlock()
	asks := append([]Fragment(nil), m.asks...)
	requests := append([]openai.ChatCompletionRequest(nil), m.requests...)
	return asks, requests, m.next, len(m.script) - m.next
}

type approvalEchoRunner struct {
	mu         sync.Mutex
	executions int
}

func (r *approvalEchoRunner) Run(map[string]any) (string, any, error) {
	r.mu.Lock()
	r.executions++
	r.mu.Unlock()
	return "echoed", nil, nil
}

func (r *approvalEchoRunner) count() int {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.executions
}

func newApprovalEchoTool(runner *approvalEchoRunner) ToolDefinitionInterface {
	return NewToolDefinition(
		runner,
		map[string]any{"type": "object", "properties": map[string]any{}},
		"approval_echo",
		"records approval test executions",
	)
}

func approvalResponse(name, args string) approvalScriptEntry {
	return approvalScriptEntry{name: name, args: args}
}

func approvalInitialScript(subtasks string) []approvalScriptEntry {
	return []approvalScriptEntry{
		approvalResponse("json", `{"extract_boolean":true}`),
		approvalResponse("json", `{"goal":"test goal"}`),
		approvalResponse("json", fmt.Sprintf(`{"subtasks":[%q]}`, subtasks)),
	}
}

func approvalSuccessfulScript(subtask string) []approvalScriptEntry {
	script := approvalInitialScript(subtask)
	return append(script,
		approvalResponse("approval_echo", `{}`),
		approvalResponse("json", `{"extract_boolean":true}`),
	)
}

func approvalInput() Fragment {
	return NewEmptyFragment().AddMessage(UserMessageRole, "test request")
}

func approvalAutomaticOptions(ctx context.Context, tool ToolDefinitionInterface, handler func(context.Context, *structures.Plan, *structures.Goal) PlanDecision) []Option {
	return []Option{
		EnableAutoPlan,
		WithTools(tool),
		WithPlanApproval(handler),
		WithContext(ctx),
		WithMaxRetries(1),
	}
}

func requireApprovalOriginalInput(t *testing.T, original Fragment) {
	t.Helper()
	if len(original.Messages) != 1 || original.Messages[0].Role != UserMessageRole.String() || original.Messages[0].Content != "test request" {
		t.Fatalf("original input mutated: %+v", original.Messages)
	}
}

func requireApprovalRejection(t *testing.T, result Fragment, err error, runner *approvalEchoRunner) {
	t.Helper()
	if !errors.Is(err, ErrPlanRejected) {
		t.Fatalf("got error %v, want ErrPlanRejected", err)
	}
	if got := runner.count(); got != 0 {
		t.Fatalf("executed %d tools after rejection", got)
	}
	last := result.LastMessage()
	if last == nil || last.Role != SystemMessageRole.String() || last.Content != "[plan rejected by user]" {
		t.Fatalf("missing rejection marker: %+v", last)
	}
}

func TestPlanApprovalApprovesOriginalBeforeExecution(t *testing.T) {
	llm := newApprovalLLM(approvalSuccessfulScript("original task")...)
	runner := &approvalEchoRunner{}
	input := approvalInput()
	decisions := 0

	result, err := ExecuteTools(llm, input, approvalAutomaticOptions(context.Background(), newApprovalEchoTool(runner), func(_ context.Context, plan *structures.Plan, goal *structures.Goal) PlanDecision {
		decisions++
		if got := runner.count(); got != 0 {
			t.Fatalf("tool ran before approval: %d", got)
		}
		if goal.Goal != "test goal" || len(plan.Subtasks) != 1 || plan.Subtasks[0] != "original task" {
			t.Fatalf("callback received goal=%+v plan=%+v", goal, plan)
		}
		return PlanDecision{Approved: true}
	})...)
	if err != nil {
		t.Fatalf("ExecuteTools: %v", err)
	}
	if decisions != 1 || runner.count() != 1 {
		t.Fatalf("decisions=%d executions=%d, want 1 each", decisions, runner.count())
	}
	asks, _, _, _ := llm.snapshot()
	if len(asks) < 4 || !strings.Contains(asks[3].String(), "original task") {
		t.Fatalf("execution prompt does not use original task: %+v", asks)
	}
	if len(result.Status.Plans) != 1 || result.Status.Plans[0].Plan.Subtasks[0] != "original task" {
		t.Fatalf("approved plan not retained in status: %+v", result.Status.Plans)
	}
}

func TestPlanApprovalUsesEditedPlanAndIgnoresApprovedFeedback(t *testing.T) {
	llm := newApprovalLLM(approvalSuccessfulScript("original task")...)
	runner := &approvalEchoRunner{}
	input := approvalInput()
	edited := &structures.Plan{Description: "edited", Subtasks: []string{"edited task"}}

	result, err := ExecuteTools(llm, input, approvalAutomaticOptions(context.Background(), newApprovalEchoTool(runner), func(_ context.Context, _ *structures.Plan, goal *structures.Goal) PlanDecision {
		if goal.Goal != "test goal" {
			t.Fatalf("goal changed: %+v", goal)
		}
		return PlanDecision{Approved: true, Plan: edited, Feedback: "ignored"}
	})...)
	if err != nil {
		t.Fatalf("ExecuteTools: %v", err)
	}
	asks, _, calls, _ := llm.snapshot()
	if calls != 5 {
		t.Fatalf("model calls=%d, want 5 without re-extraction", calls)
	}
	if len(asks) < 4 {
		t.Fatalf("recorded asks=%d, want execution prompt", len(asks))
	}
	if !strings.Contains(asks[3].String(), "edited task") || strings.Contains(asks[3].String(), "original task") {
		t.Fatalf("execution prompt does not exclusively use edited task: %s", asks[3].String())
	}
	if runner.count() != 1 || len(result.Status.Plans) != 1 || result.Status.Plans[0].Plan.Description != "edited" {
		t.Fatalf("execution/status mismatch: executions=%d plans=%+v", runner.count(), result.Status.Plans)
	}
}

func TestPlanApprovalRejectsEmptyOrWhitespaceFeedback(t *testing.T) {
	for _, feedback := range []string{"", " \n\t "} {
		t.Run(fmt.Sprintf("feedback_%q", feedback), func(t *testing.T) {
			llm := newApprovalLLM(approvalInitialScript("original task")...)
			runner := &approvalEchoRunner{}
			input := approvalInput()
			result, err := ExecuteTools(llm, input, approvalAutomaticOptions(context.Background(), newApprovalEchoTool(runner), func(context.Context, *structures.Plan, *structures.Goal) PlanDecision {
				return PlanDecision{Feedback: feedback}
			})...)
			requireApprovalRejection(t, result, err, runner)
			_, _, calls, spare := llm.snapshot()
			if calls != 3 || spare != 0 {
				t.Fatalf("model calls=%d spare=%d, want only initial extraction", calls, spare)
			}
			requireApprovalOriginalInput(t, input)
		})
	}
}

func TestPlanApprovalRevisesFromPrivateFeedbackThenExecutes(t *testing.T) {
	script := append(approvalInitialScript("original task"),
		approvalResponse("json", `{"subtasks":["revised task"]}`),
		approvalResponse("approval_echo", `{}`),
		approvalResponse("json", `{"extract_boolean":true}`),
	)
	llm := newApprovalLLM(script...)
	runner := &approvalEchoRunner{}
	input := approvalInput()
	var seen [][]string
	result, err := ExecuteTools(llm, input, approvalAutomaticOptions(context.Background(), newApprovalEchoTool(runner), func(_ context.Context, plan *structures.Plan, _ *structures.Goal) PlanDecision {
		seen = append(seen, append([]string(nil), plan.Subtasks...))
		if len(seen) == 1 {
			return PlanDecision{Feedback: "use the offline source"}
		}
		return PlanDecision{Approved: true}
	})...)
	if err != nil {
		t.Fatalf("ExecuteTools: %v", err)
	}
	if len(seen) != 2 || seen[0][0] != "original task" || seen[1][0] != "revised task" {
		t.Fatalf("callback plans: %+v", seen)
	}
	asks, _, _, _ := llm.snapshot()
	if len(asks) < 5 || !strings.Contains(asks[3].String(), "use the offline source") || !strings.Contains(asks[3].String(), "original task") {
		t.Fatalf("revision prompt omitted feedback or proposal: %+v", asks)
	}
	if !strings.Contains(asks[4].String(), "revised task") || strings.Contains(asks[4].String(), "use the offline source") {
		t.Fatalf("execution received approval transcript: %s", asks[4].String())
	}
	if runner.count() != 1 || result.Status.Plans[0].Plan.Subtasks[0] != "revised task" {
		t.Fatalf("revised plan was not executed: executions=%d plans=%+v", runner.count(), result.Status.Plans)
	}
	requireApprovalOriginalInput(t, input)
}

func TestPlanApprovalBoundsFeedbackRevisions(t *testing.T) {
	script := append(approvalInitialScript("original task"),
		approvalResponse("json", `{"subtasks":["revision one"]}`),
		approvalResponse("json", `{"subtasks":["revision two"]}`),
		approvalResponse("json", `{"subtasks":["unused revision"]}`),
	)
	llm := newApprovalLLM(script...)
	runner := &approvalEchoRunner{}
	decisions := 0
	opts := approvalAutomaticOptions(context.Background(), newApprovalEchoTool(runner), func(context.Context, *structures.Plan, *structures.Goal) PlanDecision {
		decisions++
		return PlanDecision{Feedback: "revise again"}
	})
	opts = append(opts, WithMaxAdjustmentAttempts(2))
	result, err := ExecuteTools(llm, approvalInput(), opts...)
	requireApprovalRejection(t, result, err, runner)
	_, _, calls, spare := llm.snapshot()
	if decisions != 3 || calls != 5 || spare != 1 {
		t.Fatalf("decisions=%d calls=%d spare=%d, want 3, 5, 1", decisions, calls, spare)
	}
}

func TestPlanApprovalUsesDefaultFiveRevisionBound(t *testing.T) {
	script := approvalInitialScript("original task")
	for i := 1; i <= 6; i++ {
		script = append(script, approvalResponse("json", fmt.Sprintf(`{"subtasks":["revision %d"]}`, i)))
	}
	llm := newApprovalLLM(script...)
	runner := &approvalEchoRunner{}
	decisions := 0
	result, err := ExecuteTools(llm, approvalInput(), approvalAutomaticOptions(context.Background(), newApprovalEchoTool(runner), func(context.Context, *structures.Plan, *structures.Goal) PlanDecision {
		decisions++
		return PlanDecision{Feedback: "revise again"}
	})...)
	requireApprovalRejection(t, result, err, runner)
	_, _, calls, spare := llm.snapshot()
	if decisions != 6 || calls != 8 || spare != 1 {
		t.Fatalf("decisions=%d calls=%d spare=%d, want 6, 8, 1", decisions, calls, spare)
	}
}

func TestPlanApprovalBlocksModelAndToolsUntilDecision(t *testing.T) {
	llm := newApprovalLLM(approvalSuccessfulScript("original task")...)
	runner := &approvalEchoRunner{}
	entered := make(chan struct{})
	release := make(chan struct{})
	done := make(chan error, 1)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	go func() {
		_, err := ExecuteTools(llm, approvalInput(), approvalAutomaticOptions(ctx, newApprovalEchoTool(runner), func(ctx context.Context, _ *structures.Plan, _ *structures.Goal) PlanDecision {
			close(entered)
			select {
			case <-release:
				return PlanDecision{Approved: true}
			case <-ctx.Done():
				return PlanDecision{}
			}
		})...)
		done <- err
	}()
	select {
	case <-entered:
	case <-ctx.Done():
		t.Fatal("approval callback was not reached")
	}
	_, _, calls, _ := llm.snapshot()
	if calls != 3 || runner.count() != 0 {
		t.Fatalf("work advanced while callback blocked: calls=%d executions=%d", calls, runner.count())
	}
	select {
	case err := <-done:
		t.Fatalf("ExecuteTools returned before decision: %v", err)
	default:
	}
	close(release)
	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("ExecuteTools after approval: %v", err)
		}
	case <-ctx.Done():
		t.Fatal("ExecuteTools did not resume after approval")
	}
	if runner.count() != 1 {
		t.Fatalf("executions=%d, want 1", runner.count())
	}
}

func TestPlanApprovalCancellationPrecedesCallbackDecision(t *testing.T) {
	for _, decision := range []PlanDecision{{Approved: true}, {}} {
		name := "reject"
		if decision.Approved {
			name = "approve"
		}
		t.Run(name, func(t *testing.T) {
			llm := newApprovalLLM(approvalSuccessfulScript("original task")...)
			runner := &approvalEchoRunner{}
			timeoutCtx, timeoutCancel := context.WithTimeout(context.Background(), 2*time.Second)
			defer timeoutCancel()
			ctx, cancel := context.WithCancel(timeoutCtx)
			defer cancel()
			entered := make(chan struct{})
			done := make(chan struct {
				result Fragment
				err    error
			}, 1)
			go func() {
				result, err := ExecuteTools(llm, approvalInput(), approvalAutomaticOptions(ctx, newApprovalEchoTool(runner), func(ctx context.Context, _ *structures.Plan, _ *structures.Goal) PlanDecision {
					close(entered)
					<-ctx.Done()
					return decision
				})...)
				done <- struct {
					result Fragment
					err    error
				}{result, err}
			}()
			select {
			case <-entered:
			case <-timeoutCtx.Done():
				t.Fatal("approval callback was not reached")
			}
			cancel()
			var got struct {
				result Fragment
				err    error
			}
			select {
			case got = <-done:
			case <-timeoutCtx.Done():
				t.Fatal("ExecuteTools did not return after cancellation")
			}
			if !errors.Is(got.err, context.Canceled) {
				t.Fatalf("got error %v, want context.Canceled", got.err)
			}
			if runner.count() != 0 {
				t.Fatalf("executed %d tools after cancellation", runner.count())
			}
			_, _, calls, _ := llm.snapshot()
			if calls != 3 {
				t.Fatalf("model calls=%d, want no revision or execution after cancellation", calls)
			}
			if got.result.LastMessage() == nil || got.result.LastMessage().Content != "test request" {
				t.Fatalf("cancellation did not preserve input: %+v", got.result.Messages)
			}
		})
	}
}

func TestPlanApprovalWrapsRevisionFailureAndPreservesInput(t *testing.T) {
	revisionFailure := errors.New("revision failed")
	script := append(approvalInitialScript("original task"), approvalScriptEntry{err: revisionFailure})
	llm := newApprovalLLM(script...)
	runner := &approvalEchoRunner{}
	input := approvalInput()
	result, err := ExecuteTools(llm, input, approvalAutomaticOptions(context.Background(), newApprovalEchoTool(runner), func(context.Context, *structures.Plan, *structures.Goal) PlanDecision {
		return PlanDecision{Feedback: "try another source"}
	})...)
	if !errors.Is(err, revisionFailure) {
		t.Fatalf("got error %v, want wrapped revision failure", err)
	}
	if runner.count() != 0 || len(result.Messages) != 1 || result.Messages[0].Content != "test request" {
		t.Fatalf("failure changed input or executed tool: result=%+v executions=%d", result.Messages, runner.count())
	}
	requireApprovalOriginalInput(t, input)
}

func TestPlanApprovalOptOutAndManualExecutionBoundaries(t *testing.T) {
	t.Run("nil handler keeps automatic path", func(t *testing.T) {
		llm := newApprovalLLM(approvalSuccessfulScript("original task")...)
		runner := &approvalEchoRunner{}
		_, err := ExecuteTools(llm, approvalInput(), approvalAutomaticOptions(context.Background(), newApprovalEchoTool(runner), nil)...)
		if err != nil || runner.count() != 1 {
			t.Fatalf("nil handler changed automatic execution: err=%v executions=%d", err, runner.count())
		}
		_, _, calls, _ := llm.snapshot()
		if calls != 5 {
			t.Fatalf("nil handler added model calls: %d", calls)
		}
	})

	t.Run("declined planning does not call handler", func(t *testing.T) {
		llm := newApprovalLLM(approvalResponse("json", `{"extract_boolean":false}`))
		called := false
		result, planned, err := doPlan(llm, approvalInput(), nil, WithPlanApproval(func(context.Context, *structures.Plan, *structures.Goal) PlanDecision {
			called = true
			return PlanDecision{Approved: true}
		}))
		if err != nil || planned || called || result.LastMessage().Content != "test request" {
			t.Fatalf("declined plan boundary: err=%v planned=%v called=%v result=%+v", err, planned, called, result.Messages)
		}
	})

	t.Run("direct ExecutePlan remains caller managed", func(t *testing.T) {
		llm := newApprovalLLM(
			approvalResponse("approval_echo", `{}`),
			approvalResponse("json", `{"extract_boolean":true}`),
		)
		runner := &approvalEchoRunner{}
		called := false
		plan := &structures.Plan{Description: "manual", Subtasks: []string{"manual task"}}
		result, err := ExecutePlan(llm, approvalInput(), plan, &structures.Goal{Goal: "manual goal"},
			WithTools(newApprovalEchoTool(runner)),
			WithMaxRetries(1),
			WithPlanApproval(func(context.Context, *structures.Plan, *structures.Goal) PlanDecision {
				called = true
				return PlanDecision{}
			}),
		)
		if err != nil || called || runner.count() != 1 || len(result.Status.Plans) != 1 {
			t.Fatalf("manual plan was gated: err=%v called=%v executions=%d plans=%+v", err, called, runner.count(), result.Status.Plans)
		}
	})
}

func TestPlanApprovalRejectsEmptyEditedPlanWithoutExecution(t *testing.T) {
	llm := newApprovalLLM(approvalInitialScript("original task")...)
	runner := &approvalEchoRunner{}
	input := approvalInput()
	result, err := ExecuteTools(llm, input, approvalAutomaticOptions(context.Background(), newApprovalEchoTool(runner), func(context.Context, *structures.Plan, *structures.Goal) PlanDecision {
		return PlanDecision{Approved: true, Plan: &structures.Plan{Description: "empty"}}
	})...)
	if err == nil || !strings.Contains(err.Error(), "approved plan has no subtasks") {
		t.Fatalf("got error %v, want useful empty-plan error", err)
	}
	if runner.count() != 0 || len(result.Messages) != 1 || result.Messages[0].Content != "test request" {
		t.Fatalf("invalid edit changed input or executed tool: result=%+v executions=%d", result.Messages, runner.count())
	}
	requireApprovalOriginalInput(t, input)
}

func TestPlanApprovalAccumulatesFeedbackPrivately(t *testing.T) {
	script := append(approvalInitialScript("original task"),
		approvalResponse("json", `{"subtasks":["revision one"]}`),
		approvalResponse("json", `{"subtasks":["revision two"]}`),
		approvalResponse("approval_echo", `{}`),
		approvalResponse("json", `{"extract_boolean":true}`),
	)
	llm := newApprovalLLM(script...)
	runner := &approvalEchoRunner{}
	input := approvalInput()
	decision := 0
	result, err := ExecuteTools(llm, input, approvalAutomaticOptions(context.Background(), newApprovalEchoTool(runner), func(context.Context, *structures.Plan, *structures.Goal) PlanDecision {
		decision++
		switch decision {
		case 1:
			return PlanDecision{Feedback: "first feedback"}
		case 2:
			return PlanDecision{Feedback: "second feedback"}
		default:
			return PlanDecision{Approved: true}
		}
	})...)
	if err != nil {
		t.Fatalf("ExecuteTools: %v", err)
	}
	asks, _, _, _ := llm.snapshot()
	if len(asks) < 6 {
		t.Fatalf("recorded asks=%d, want revision and execution prompts", len(asks))
	}
	secondRevision := asks[4].String()
	for _, want := range []string{"first feedback", "second feedback", "revision one"} {
		if !strings.Contains(secondRevision, want) {
			t.Fatalf("second revision prompt omitted %q: %s", want, secondRevision)
		}
	}
	executionPrompt := asks[5].String()
	if !strings.Contains(executionPrompt, "revision two") || strings.Contains(executionPrompt, "first feedback") || strings.Contains(executionPrompt, "second feedback") {
		t.Fatalf("execution context leaked feedback transcript: %s", executionPrompt)
	}
	if runner.count() != 1 || result.Status.Plans[0].Plan.Subtasks[0] != "revision two" {
		t.Fatalf("latest revision not executed: executions=%d plans=%+v", runner.count(), result.Status.Plans)
	}
	requireApprovalOriginalInput(t, input)
}
