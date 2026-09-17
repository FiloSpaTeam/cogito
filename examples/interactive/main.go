// Command interactive demonstrates Cogito's interactive embedding primitives
// with a deterministic, offline model and dispatcher. It needs no API key or
// network access.
package main

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/signal"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/mudler/cogito"
	"github.com/mudler/cogito/structures"
	"github.com/mudler/xlog"
	"github.com/sashabaranov/go-openai"
)

func main() {
	xlog.SetLogger(xlog.NewLogger(xlog.LogLevel("error"), ""))
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	if err := run(ctx, os.Stdin, os.Stdout); err != nil {
		if errors.Is(err, context.Canceled) {
			fmt.Fprintln(os.Stderr, "Interactive demo cancelled.")
			return
		}
		fmt.Fprintln(os.Stderr, "Interactive demo:", err)
		os.Exit(1)
	}
}

func run(ctx context.Context, in io.Reader, out io.Writer) error {
	if err := ctx.Err(); err != nil {
		return err
	}

	printer := &lockedWriter{w: out}
	input := newLineInput(ctx, in)
	defer input.stop()

	printer.printf("Cogito interactive demo (offline scripted model and dispatcher)\n")
	fragment := cogito.NewEmptyFragment().AddMessage(cogito.UserMessageRole, "Help me prepare a local verification report.")

	var err error
	fragment, err = questionStage(ctx, input, printer, fragment)
	if err != nil {
		return err
	}

	var approved bool
	fragment, approved, err = planStage(ctx, input, printer, fragment)
	if err != nil {
		return err
	}
	if !approved {
		return nil
	}

	_, err = backgroundStage(ctx, printer, fragment)
	return err
}

type lockedWriter struct {
	mu sync.Mutex
	w  io.Writer
}

func (w *lockedWriter) printf(format string, args ...any) {
	w.mu.Lock()
	defer w.mu.Unlock()
	fmt.Fprintf(w.w, format, args...)
}

type inputEvent struct {
	line string
	err  error
}

// lineInput is the demo's only reader goroutine. Every interactive callback
// selects on the same channel, so EOF and cancellation release blocked stages.
// A reader that can block indefinitely must also implement io.Closer (as
// os.Stdin and io.PipeReader do) so stop can join the goroutine on cancellation.
type lineInput struct {
	cancel context.CancelFunc
	done   chan struct{}
	events chan inputEvent
	closer io.Closer
	once   sync.Once
}

func newLineInput(parent context.Context, r io.Reader) *lineInput {
	ctx, cancel := context.WithCancel(parent)
	in := &lineInput{
		cancel: cancel,
		done:   make(chan struct{}),
		events: make(chan inputEvent, 1),
	}
	if closer, ok := r.(io.Closer); ok {
		in.closer = closer
	}

	go func() {
		defer close(in.done)
		scanner := bufio.NewScanner(r)
		for scanner.Scan() {
			select {
			case in.events <- inputEvent{line: scanner.Text()}:
			case <-ctx.Done():
				return
			}
		}
		err := scanner.Err()
		if err == nil {
			err = io.EOF
		}
		select {
		case in.events <- inputEvent{err: err}:
		case <-ctx.Done():
		}
	}()
	return in
}

func (in *lineInput) next(ctx context.Context) (string, error) {
	select {
	case <-ctx.Done():
		return "", ctx.Err()
	case event := <-in.events:
		return strings.TrimSpace(event.line), event.err
	}
}

func (in *lineInput) stop() {
	in.once.Do(func() {
		in.cancel()
		if in.closer != nil {
			_ = in.closer.Close()
		}
		<-in.done
	})
}

type stageResult struct {
	fragment cogito.Fragment
	err      error
}

func questionStage(ctx context.Context, input *lineInput, out *lockedWriter, fragment cogito.Fragment) (cogito.Fragment, error) {
	stageCtx, cancel := context.WithCancel(ctx)
	defer cancel()

	questions := make(chan cogito.UserQuestion, 1)
	registry := cogito.NewQuestionRegistry(func(q cogito.UserQuestion) {
		select {
		case questions <- q:
		case <-stageCtx.Done():
		}
	})
	result := make(chan stageResult, 1)
	go func() {
		f, err := cogito.ExecuteTools(
			&scriptedModel{stage: scriptQuestion},
			fragment,
			cogito.WithContext(stageCtx),
			cogito.WithUserQuestions(registry.Handle),
			cogito.WithIterations(3),
		)
		result <- stageResult{fragment: f, err: err}
	}()

	for {
		select {
		case <-ctx.Done():
			cancel()
			joinStage(result)
			return fragment, ctx.Err()
		case r := <-result:
			return r.fragment, r.err
		case q := <-questions:
			pending := registry.Pending()
			if len(pending) == 0 {
				cancel()
				return fragment, errors.New("question notification had no pending registry entry")
			}
			out.printf("\nQuestion: %s\n", q.Question)
			out.printf("Options: %s (or text:YOUR ANSWER)\n> ", strings.Join(q.Options, ", "))
			for {
				line, err := input.next(stageCtx)
				if err != nil {
					cancel()
					joinStage(result)
					return fragment, err
				}
				answer := parseAnswer(q, line)
				if err := registry.Answer(q.ID, answer); err != nil {
					out.printf("Invalid answer: %v\n> ", err)
					continue
				}
				out.printf("Question answer: %s\n", displayAnswer(answer))
				break
			}
		}
	}
}

func joinStage(result <-chan stageResult) {
	select {
	case <-result:
	case <-time.After(time.Second):
	}
}

func parseAnswer(q cogito.UserQuestion, line string) cogito.UserAnswer {
	if text, ok := strings.CutPrefix(line, "text:"); ok {
		return cogito.UserAnswer{Text: strings.TrimSpace(text)}
	}
	parts := strings.Split(line, ",")
	selected := make([]string, 0, len(parts))
	for _, part := range parts {
		candidate := strings.TrimSpace(part)
		found := false
		for _, option := range q.Options {
			if candidate == option {
				selected = append(selected, candidate)
				found = true
				break
			}
		}
		if !found {
			return cogito.UserAnswer{Text: line}
		}
	}
	return cogito.UserAnswer{Selected: selected}
}

func displayAnswer(answer cogito.UserAnswer) string {
	if len(answer.Selected) > 0 {
		return strings.Join(answer.Selected, ", ")
	}
	return answer.Text
}

type recordPlanArgs struct {
	Summary string `json:"summary" description:"Short description of the approved plan step"`
}

type recordPlanTool struct{ out *lockedWriter }

func (t *recordPlanTool) Run(args recordPlanArgs) (string, any, error) {
	t.out.printf("Plan step executed: %s\n", args.Summary)
	return "approved plan step recorded", args, nil
}

func planStage(ctx context.Context, input *lineInput, out *lockedWriter, fragment cogito.Fragment) (cogito.Fragment, bool, error) {
	tool := cogito.NewToolDefinition(
		&recordPlanTool{out: out},
		recordPlanArgs{},
		"record_plan",
		"Record the approved demonstration plan locally.",
	)

	var inputErr error
	revision := 0
	approval := func(callbackCtx context.Context, plan *structures.Plan, goal *structures.Goal) cogito.PlanDecision {
		label := "Proposed plan"
		if revision > 0 {
			label = "Revised plan"
		}
		out.printf("\n%s: %s\n", label, plan.Description)
		for i, step := range plan.Subtasks {
			out.printf("  %d. %s\n", i+1, step)
		}
		out.printf("Decision [approve | reject | feedback:TEXT | edit:DESCRIPTION | STEP]: ")

		for {
			line, err := input.next(callbackCtx)
			if err != nil {
				inputErr = err
				return cogito.PlanDecision{}
			}
			switch {
			case line == "approve":
				out.printf("Approved plan: %s\n", plan.Description)
				return cogito.PlanDecision{Approved: true}
			case line == "reject":
				return cogito.PlanDecision{}
			case strings.HasPrefix(line, "feedback:"):
				feedback := strings.TrimSpace(strings.TrimPrefix(line, "feedback:"))
				if feedback == "" {
					out.printf("Feedback must not be empty.\n> ")
					continue
				}
				revision++
				return cogito.PlanDecision{Feedback: feedback}
			case strings.HasPrefix(line, "edit:"):
				parts := splitNonEmpty(strings.TrimPrefix(line, "edit:"), "|")
				if len(parts) < 2 {
					out.printf("Edit syntax is edit:DESCRIPTION | STEP [| STEP].\n> ")
					continue
				}
				edited := *plan
				edited.Description = parts[0]
				edited.Subtasks = append([]string(nil), parts[1:]...)
				out.printf("Approved plan: %s\n", edited.Description)
				return cogito.PlanDecision{Approved: true, Plan: &edited}
			default:
				out.printf("Enter approve, reject, feedback:TEXT, or edit:DESCRIPTION | STEP.\n> ")
			}
		}
	}

	request := fragment.AddMessage(cogito.UserMessageRole, "Create and execute a one-step offline verification plan.")
	result, err := cogito.ExecuteTools(
		&scriptedModel{stage: scriptPlan},
		request,
		cogito.WithContext(ctx),
		cogito.WithTools(tool),
		cogito.EnableAutoPlan,
		cogito.WithPlanApproval(approval),
		cogito.WithMaxAdjustmentAttempts(3),
		cogito.WithIterations(4),
	)
	if inputErr != nil {
		return result, false, inputErr
	}
	if errors.Is(err, cogito.ErrPlanRejected) {
		out.printf("Plan rejected; no work was executed.\n")
		return result, false, nil
	}
	if err != nil {
		return result, false, err
	}
	return result, true, nil
}

func splitNonEmpty(text, separator string) []string {
	var result []string
	for _, part := range strings.Split(text, separator) {
		if part = strings.TrimSpace(part); part != "" {
			result = append(result, part)
		}
	}
	return result
}

func backgroundStage(ctx context.Context, out *lockedWriter, fragment cogito.Fragment) (cogito.Fragment, error) {
	manager := cogito.NewAgentManager()
	injections := make(chan openai.ChatCompletionMessage, 4)
	parked := make(chan struct{}, 1)

	dispatcher := func(workerCtx context.Context, spec cogito.AgentRunSpec) (cogito.Fragment, error) {
		if spec.Emit != nil {
			spec.Emit(cogito.AgentEvent{AgentID: spec.ID, Kind: "running", Delta: "offline dispatcher started"})
		}
		select {
		case <-parked:
		case <-workerCtx.Done():
			return cogito.Fragment{}, workerCtx.Err()
		}
		return cogito.NewEmptyFragment().AddMessage(
			cogito.AssistantMessageRole,
			"go test ./... and go vet ./...",
		), nil
	}

	request := fragment.AddMessage(cogito.UserMessageRole, "Delegate the final verification report to the demo reporter.")
	result, err := cogito.ExecuteTools(
		&scriptedModel{stage: scriptBackground},
		request,
		cogito.WithContext(ctx),
		cogito.EnableAgentSpawning,
		cogito.WithAgentManager(manager),
		cogito.WithAgentDefinitions(cogito.AgentDefinition{
			Name:         "demo-reporter",
			Description:  "Produces a deterministic local verification report",
			SystemPrompt: "Return the scripted local verification report.",
		}),
		cogito.WithAgentDispatcher(dispatcher),
		cogito.WithMessageInjectionChan(injections),
		cogito.WithAgentCompletionFormatter(func(agent *cogito.AgentState) string {
			return "[demo completion] " + agent.Result
		}),
		cogito.WithOnPark(func(reply string) {
			out.printf("Parked reply: %s\n", reply)
			select {
			case parked <- struct{}{}:
			default:
			}
		}),
		cogito.WithOnResume(func() {
			out.printf("Resumed after injected completion.\n")
		}),
		cogito.WithStreamCallback(func(event cogito.StreamEvent) {
			if event.Type == cogito.StreamEventSubAgent && event.AgentStatus != "" {
				out.printf("Agent status: %s\n", event.AgentStatus)
			}
		}),
		cogito.WithIterations(6),
	)

	cleanupErr := cancelAndJoinBackground(manager)
	if err != nil {
		return result, err
	}
	if cleanupErr != nil {
		return result, cleanupErr
	}
	out.printf("%s\n", result.LastMessage().Content)
	return result, nil
}

func cancelAndJoinBackground(manager *cogito.AgentManager) error {
	agents := manager.List()
	for _, agent := range agents {
		if agent.Cancel != nil {
			agent.Cancel()
		}
	}
	for _, agent := range agents {
		waited := make(chan error, 1)
		go func(id string) {
			_, err := manager.Wait(id)
			waited <- err
		}(agent.ID)
		select {
		case err := <-waited:
			if err != nil {
				return err
			}
		case <-time.After(time.Second):
			return fmt.Errorf("timed out joining background agent %s", agent.ID)
		}
	}
	return nil
}

type scriptStage uint8

const (
	scriptQuestion scriptStage = iota
	scriptPlan
	scriptBackground
)

// scriptedModel is deliberately semantic rather than request-count driven: it
// selects a response from the offered tools, requested JSON schema, and actual
// conversation. That keeps approval feedback and retries deterministic.
type scriptedModel struct {
	stage scriptStage
	mu    sync.Mutex
	next  int
}

func (m *scriptedModel) Ask(ctx context.Context, fragment cogito.Fragment) (cogito.Fragment, error) {
	if err := ctx.Err(); err != nil {
		return fragment, err
	}
	last := ""
	if message := fragment.LastMessage(); message != nil {
		last = message.Content
	}
	reply := "Scripted stage complete."
	if m.stage == scriptPlan {
		switch {
		case strings.Contains(last, "decides if planning"):
			reply = "yes"
		case strings.Contains(last, "Analyze the following text"):
			reply = "Prepare an offline command report"
		case strings.Contains(last, "breaks down a goal"):
			if strings.Contains(last, "keep one concise step") {
				reply = "Prepare one concise offline command report"
			} else {
				reply = "Prepare an offline command report"
			}
		case strings.Contains(last, "determines if a goal has been achieved"):
			reply = "yes"
		}
	}
	return fragment.AddMessage(cogito.AssistantMessageRole, reply), nil
}

func (m *scriptedModel) CreateChatCompletion(ctx context.Context, request openai.ChatCompletionRequest) (cogito.LLMReply, cogito.LLMUsage, error) {
	if err := ctx.Err(); err != nil {
		return cogito.LLMReply{}, cogito.LLMUsage{}, err
	}
	if hasOfferedTool(request, "json") {
		return m.toolReply("json", jsonArguments(request)), cogito.LLMUsage{}, nil
	}

	switch m.stage {
	case scriptQuestion:
		if !hasCalledTool(request.Messages, cogito.UserQuestionToolName) {
			return m.toolReply(cogito.UserQuestionToolName, `{"question":"Which kind of help should the demo produce?","options":["print commands","explain concepts"],"allow_free_text":true}`), cogito.LLMUsage{}, nil
		}
		return textReply("Question answer received."), cogito.LLMUsage{}, nil
	case scriptPlan:
		if hasOfferedTool(request, "record_plan") && !hasCalledTool(request.Messages, "record_plan") {
			return m.toolReply("record_plan", `{"summary":"run local Go checks"}`), cogito.LLMUsage{}, nil
		}
		return textReply("Approved plan execution complete."), cogito.LLMUsage{}, nil
	case scriptBackground:
		if hasMessage(request.Messages, "[demo completion]") {
			return textReply("Final background report: go test ./... and go vet ./..."), cogito.LLMUsage{}, nil
		}
		if !hasCalledTool(request.Messages, "spawn_agent") {
			return m.toolReply("spawn_agent", `{"agent_type":"demo-reporter","task":"Report the local verification commands","background":true}`), cogito.LLMUsage{}, nil
		}
		return textReply("Background worker launched; waiting for its report."), cogito.LLMUsage{}, nil
	default:
		return textReply("Scripted stage complete."), cogito.LLMUsage{}, nil
	}
}

func (m *scriptedModel) toolReply(name, arguments string) cogito.LLMReply {
	m.mu.Lock()
	m.next++
	id := fmt.Sprintf("scripted-%d-%d", m.stage, m.next)
	m.mu.Unlock()
	return cogito.LLMReply{ChatCompletionResponse: openai.ChatCompletionResponse{
		Choices: []openai.ChatCompletionChoice{{Message: openai.ChatCompletionMessage{
			Role: cogito.AssistantMessageRole.String(),
			ToolCalls: []openai.ToolCall{{
				ID:       id,
				Type:     openai.ToolTypeFunction,
				Function: openai.FunctionCall{Name: name, Arguments: arguments},
			}},
		}}},
	}}
}

func textReply(content string) cogito.LLMReply {
	return cogito.LLMReply{ChatCompletionResponse: openai.ChatCompletionResponse{
		Choices: []openai.ChatCompletionChoice{{Message: openai.ChatCompletionMessage{
			Role:    cogito.AssistantMessageRole.String(),
			Content: content,
		}}},
	}}
}

func jsonArguments(request openai.ChatCompletionRequest) string {
	var parameters any
	for _, tool := range request.Tools {
		if tool.Function != nil && tool.Function.Name == "json" {
			parameters = tool.Function.Parameters
			break
		}
	}
	encoded, _ := json.Marshal(parameters)
	var schema struct {
		Properties map[string]json.RawMessage `json:"properties"`
	}
	_ = json.Unmarshal(encoded, &schema)
	switch {
	case hasProperty(schema.Properties, "extract_boolean"):
		return `{"extract_boolean":true}`
	case hasProperty(schema.Properties, "goal"):
		return `{"goal":"Prepare an offline command report"}`
	case hasProperty(schema.Properties, "subtasks"):
		return `{"description":"offline demo plan","subtasks":["Use record_plan to record the local verification commands"]}`
	default:
		return `{}`
	}
}

func hasProperty(properties map[string]json.RawMessage, name string) bool {
	_, ok := properties[name]
	return ok
}

func hasOfferedTool(request openai.ChatCompletionRequest, name string) bool {
	for _, tool := range request.Tools {
		if tool.Function != nil && tool.Function.Name == name {
			return true
		}
	}
	return false
}

func hasCalledTool(messages []openai.ChatCompletionMessage, name string) bool {
	for _, message := range messages {
		for _, call := range message.ToolCalls {
			if call.Function.Name == name {
				return true
			}
		}
	}
	return false
}

func hasMessage(messages []openai.ChatCompletionMessage, text string) bool {
	for _, message := range messages {
		if strings.Contains(message.Content, text) {
			return true
		}
	}
	return false
}
